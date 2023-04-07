import math
from typing import Dict, Optional

import pytorch_lightning as pl
import torch
from einops import rearrange
from matplotlib import pyplot as plt
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
from torch import Tensor
from torch.distributions import Distribution
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
from yolov5.models.yolo import DetectionModel

from bliss.catalog import TileCatalog
from bliss.metrics import DetectionMetrics
from bliss.plotting import plot_detections
from bliss.unconstrained_dists import (
    TransformedBernoulli,
    TransformedDiagonalBivariateNormal,
    TransformedLogitNormal,
    TransformedLogNormal,
    TransformedNormal,
)


class Encoder(pl.LightningModule):
    """Encodes the distribution of a latent variable representing an astronomical image.

    This class implements the source encoder, which is supposed to take in
    an astronomical image of size slen * slen and returns a NN latent variable
    representation of this image.
    """

    def __init__(
        self,
        architecture: DictConfig,
        n_bands: int,
        tile_slen: int,
        tiles_to_crop: int,
        annotate_probs: bool = False,
        slack: float = 1.0,
        optimizer_params: Optional[dict] = None,
        scheduler_params: Optional[dict] = None,
    ):
        """Initializes DetectionEncoder.

        Args:
            architecture: yaml to specifying the encoder network architecture
            n_bands: number of bands
            tile_slen: dimension in pixels of a square tile
            tiles_to_crop: margin of tiles not to use for computing loss
            annotate_probs: Annotate probabilities on validation plots?
            slack: Slack to use when matching locations for validation metrics.
            optimizer_params: arguments passed to the Adam optimizer
            scheduler_params: arguments passed to the learning rate scheduler
        """
        super().__init__()

        self.n_bands = n_bands
        self.optimizer_params = optimizer_params
        self.scheduler_params = scheduler_params if scheduler_params else {"milestones": []}

        self.tile_slen = tile_slen

        # number of distributional parameters used to characterize each source
        self.n_params_per_source = sum(param.dim for param in self.dist_param_groups.values())

        # a hack to get the right number of outputs from yolo
        architecture["nc"] = self.n_params_per_source - 5
        arch_dict = OmegaConf.to_container(architecture)
        self.model = DetectionModel(cfg=arch_dict, ch=2)
        self.tiles_to_crop = tiles_to_crop

        # plotting
        self.annotate_probs = annotate_probs

        # metrics
        self.metrics = DetectionMetrics(slack)

    @property
    def dist_param_groups(self):
        # TODO: override this method in case_studies/strong_lensing to detect strong lenses too
        return {
            "on_prob": TransformedBernoulli(),
            "loc": TransformedDiagonalBivariateNormal(),
            "star_log_flux": TransformedNormal(low_clamp=-6, high_clamp=3),
            "galaxy_prob": TransformedBernoulli(),
            # galsim parameters
            "galsim_flux": TransformedLogNormal(),
            "galsim_disk_frac": TransformedLogitNormal(),
            "galsim_beta_radians": TransformedLogitNormal(high=2 * torch.pi),
            "galsim_disk_q": TransformedLogitNormal(),
            "galsim_a_d": TransformedLogNormal(),
            "galsim_bulge_q": TransformedLogitNormal(),
            "galsim_a_b": TransformedLogNormal(),
        }

    def encode_batch(self, batch):
        # TODO: consider adding multiple log and sigmoid transformed images to this stack,
        # each with a different offset and scale
        images_with_background = torch.cat((batch["images"], batch["background"]), dim=1)

        # setting this to true every time is a hack to make yolo DetectionModel
        # give us output of the right dimension
        self.model.model[-1].training = True

        output = self.model(images_with_background)
        # there's an extra dimension for channel that is always a singleton
        output4d = rearrange(output[0], "b 1 ht hw pps -> b ht hw pps")

        ttc = self.tiles_to_crop
        output_cropped = output4d[:, ttc:-ttc, ttc:-ttc, :]

        split_sizes = [v.dim for v in self.dist_param_groups.values()]
        dist_params_split = torch.split(output_cropped, split_sizes, 3)
        names = self.dist_param_groups.keys()
        pred = dict(zip(names, dist_params_split))

        for k, v in pred.items():
            pred[k] = self.dist_param_groups[k].get_dist(v)

        return pred

    def variational_mode(self, pred: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """Compute the mode of the variational distribution."""
        # TODO: the mean would be better at minimizing squared error...return that instead?
        tile_is_on_array = pred["on_prob"].mode
        # this is the mode of star_log_flux but not the mean of the star_flux distribution
        star_fluxes = pred["star_log_flux"].mode.exp()  # type: ignore
        star_fluxes *= tile_is_on_array

        # we have to unsqueeze some tensors below because a TileCatalog can store multiple
        # light sources per tile, but we predict only one source per tile
        est_catalog_dict = {
            "locs": rearrange(pred["loc"].mode, "b ht wt d -> b ht wt 1 d"),
            "star_log_fluxes": rearrange(pred["star_log_flux"].mode, "b ht wt -> b ht wt 1 1"),
            "star_fluxes": rearrange(star_fluxes, "b ht wt -> b ht wt 1 1"),
            "n_sources": tile_is_on_array,
        }
        est_tile_catalog = TileCatalog(self.tile_slen, est_catalog_dict)
        return est_tile_catalog.to_full_params()

    def configure_optimizers(self):
        """Configure optimizers for training (pytorch lightning)."""
        optimizer = Adam(self.parameters(), **self.optimizer_params)
        # TODO: tune this more and try the OneCycle learning rate scheduler
        scheduler = MultiStepLR(optimizer, **self.scheduler_params)
        return [optimizer], [scheduler]

    def _get_loss(self, pred: Dict[str, Distribution], true_tile_cat: TileCatalog):
        # counter loss
        counter_loss = -pred["on_prob"].log_prob(true_tile_cat.n_sources)

        # location loss
        # all the squeezing/rearranging below is because a TileCatalog can store multiple
        # light sources per tile, which is annoying here, but helpful for storing samples
        # real catlogs. Still, we may want to change it to just store one.
        locs_loss = -pred["loc"].log_prob(true_tile_cat.locs.squeeze(3)).sum(-1)
        locs_loss *= true_tile_cat.is_on_array.squeeze(3)

        # star flux loss
        star_log_fluxes = rearrange(true_tile_cat["star_log_fluxes"], "b ht wt 1 1 -> b ht wt")
        star_flux_loss = -pred["star_log_flux"].log_prob(star_log_fluxes)
        star_flux_loss *= true_tile_cat.is_on_array.squeeze(3)
        true_gal_bools = rearrange(true_tile_cat["galaxy_bools"], "b ht wt 1 1 -> b ht wt")
        star_flux_loss *= 1 - true_gal_bools

        # star/galaxy classification loss
        binary_loss = -pred["galaxy_prob"].log_prob(true_gal_bools.long())
        binary_loss *= true_tile_cat.is_on_array.squeeze(3)

        # TODO: compute galsim loss
        # TODO: consider using a different NN head (but same backbone) for galsim parameters
        # TODO: explore whether small tiles are helpful for galsim parameter estimation

        loss = counter_loss + locs_loss + star_flux_loss + binary_loss

        return {
            "loss": loss.mean(),
            "counter_loss": counter_loss.mean(),
            "locs_loss": locs_loss.mean(),
            "star_flux_loss": star_flux_loss.mean(),
            "binary_loss": binary_loss.mean(),
        }

    def _generic_step(self, batch, logging_name, log_metrics=False, plot_images=False):
        batch_size = len(batch["n_sources"])
        pred = self.encode_batch(batch)
        true_tile_catalog = TileCatalog(self.tile_slen, batch)
        loss_dict = self._get_loss(pred, true_tile_catalog)
        true_full_cat = true_tile_catalog.to_full_params()
        est_cat = self.variational_mode(pred)

        # log all losses
        for k, v in loss_dict.items():
            self.log("{}/{}".format(logging_name, k), v, batch_size=batch_size)

        # log all metrics
        if log_metrics:
            # TODO: figure out why these metrics so slow to compute
            metrics = self.metrics(true_full_cat, est_cat)
            for k, v in metrics.items():
                self.log("{}/{}".format(logging_name, k), v, batch_size=batch_size)

        # log a grid of figures to the tensorboard
        if plot_images:
            batch_size = len(batch["images"])
            n_samples = min(int(math.sqrt(batch_size)) ** 2, 16)
            nrows = int(n_samples**0.5)  # for figure
            wrong_idx = (est_cat.n_sources != true_full_cat.n_sources).nonzero()
            wrong_idx = wrong_idx.view(-1)[:n_samples]
            margin_px = self.tiles_to_crop * self.tile_slen
            # TODO: clean up the plot_detection interface
            fig = plot_detections(
                batch["images"], true_full_cat, est_cat, nrows, wrong_idx, margin_px
            )
            title_root = f"Epoch:{self.current_epoch}/"
            title = f"{title_root}{logging_name} images"
            self.logger.experiment.add_figure(title, fig)
            plt.close(fig)

        return loss_dict["loss"]

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        """Training step (pytorch lightning)."""
        return self._generic_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        """Pytorch lightning method."""
        self._generic_step(batch, "val", log_metrics=True, plot_images=True)
        return batch  # do we really need to save all these batches?

    def validation_epoch_end(self, outputs):
        """Pytorch lightning method."""
        batch: Dict[str, Tensor] = outputs[-1]
        self._generic_step(batch, "val")

    def test_step(self, batch, batch_idx):
        """Pytorch lightning method."""
        self._generic_step(batch, "test", log_metrics=True)
        return batch  # do we really need to save all these batches?
