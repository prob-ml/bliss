import math
from typing import Dict, Optional, Union

import pytorch_lightning as pl
import torch
from einops import rearrange
from matplotlib import pyplot as plt
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
from torch import Tensor
from torch.distributions import Categorical, Normal
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
from yolov5.models.yolo import DetectionModel

from bliss.catalog import TileCatalog
from bliss.metrics import DetectionMetrics
from bliss.plotting import plot_detections


class DetectionEncoder(pl.LightningModule):
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
        self.n_params_per_source = sum(param["dim"] for param in self.dist_param_groups.values())

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
        return {
            "on_prob": {"dim": 1},
            "loc_mean": {"dim": 2},
            "loc_logvar": {"dim": 2},
            "log_flux_mean": {"dim": self.n_bands},
            "log_flux_logvar": {"dim": self.n_bands},
            "galaxy_prob": {"dim": 1},
        }

    def encode_batch(self, batch):
        images_with_background = torch.cat((batch["images"], batch["background"]), dim=1)

        # setting this to true every time is a hack to make yolo DetectionModel
        # give us output of the right dimension
        self.model.model[-1].training = True

        output = self.model(images_with_background)
        # there's an extra dimension for channel that is always a singleton
        output4d = rearrange(output[0], "b 1 ht hw pps -> b ht hw pps")

        ttc = self.tiles_to_crop
        output_cropped = output4d[:, ttc:-ttc, ttc:-ttc, :]

        split_sizes = [v["dim"] for v in self.dist_param_groups.values()]
        dist_params_split = torch.split(output_cropped, split_sizes, 3)
        names = self.dist_param_groups.keys()
        pred = dict(zip(names, dist_params_split))

        # should verify that these clamps aren't too extreme
        pred["on_prob"] = pred["on_prob"].sigmoid().clamp(1e-3, 1 - 1e-3)
        pred["loc_mean"] = pred["loc_mean"].sigmoid()
        pred["loc_sd"] = pred["loc_logvar"].clamp(-6, 3).exp().sqrt()
        pred["log_flux_sd"] = pred["log_flux_logvar"].clamp(-6, 10).exp().sqrt()
        pred["galaxy_prob"] = pred["galaxy_prob"].sigmoid().clamp(1e-3, 1 - 1e-3)

        # delete these so we don't accidentally use them
        del pred["loc_logvar"]
        del pred["log_flux_logvar"]

        return pred

    def sample(
        self,
        pred: Dict[str, Tensor],
        n_samples: Union[int, None],
    ) -> Dict[str, Tensor]:
        """Sample from the encoded variational distribution.

        Args:
            pred:
                The distributional parameters in matrix form.
            n_samples:
                The number of samples to draw. If None, the variational mode is taken instead.

        Returns:
            A dictionary of tensors with shape `n_samples * n_ptiles * max_sources * ...`.
            Consists of `"n_sources", "locs", "star_log_fluxes", and "star_fluxes"`.
        """
        off_on_prob = torch.cat([1 - pred["on_prob"], pred["on_prob"]], dim=1)
        tile_is_on_array = Categorical(probs=off_on_prob).sample((n_samples,))

        tile_locs = Normal(pred["loc_mean"], pred["loc_sd"]).rsample()
        tile_log_fluxes = Normal(pred["log_flux_mean"], pred["log_flux_sd"]).rsample()

        tile_fluxes = tile_log_fluxes.exp()
        tile_fluxes *= tile_is_on_array

        return {
            "locs": tile_locs,
            "star_log_fluxes": tile_log_fluxes,
            "star_fluxes": tile_fluxes,
            "n_sources": tile_is_on_array,
        }

    def variational_mode(self, pred: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """Compute the mode of the variational distribution."""
        tile_is_on_array = (pred["on_prob"] > 0.5).int()
        tile_fluxes = pred["log_flux_mean"].exp()
        tile_fluxes *= tile_is_on_array

        # we have to unsqueeze some tensors below because a TileCatalog can store multiple
        # light sources per tile, but we predict only one source per tile
        est_catalog_dict = {
            "locs": rearrange(pred["loc_mean"], "b ht wt d -> b ht wt 1 d"),
            "star_log_fluxes": rearrange(pred["log_flux_mean"], "b ht wt d -> b ht wt 1 d"),
            "star_fluxes": rearrange(tile_fluxes, "b ht wt d -> b ht wt 1 d"),
            "n_sources": rearrange(tile_is_on_array, "b ht wt 1 -> b ht wt"),
        }
        est_tile_catalog = TileCatalog(self.tile_slen, est_catalog_dict)
        return est_tile_catalog.to_full_params()

    def configure_optimizers(self):
        """Configure optimizers for training (pytorch lightning)."""
        optimizer = Adam(self.parameters(), **self.optimizer_params)
        scheduler = MultiStepLR(optimizer, **self.scheduler_params)
        return [optimizer], [scheduler]

    def _get_loss(self, pred: Dict[str, Tensor], true_tile_cat: TileCatalog):
        # counter loss
        on_prob_flat = rearrange(pred["on_prob"], "b ht wt 1 -> (b ht wt) 1")
        off_on_prob = torch.cat([1 - on_prob_flat, on_prob_flat], dim=1)
        true_on_flat = true_tile_cat.n_sources.reshape(-1)
        counter_loss = -Categorical(off_on_prob).log_prob(true_on_flat)

        # location loss
        loc_dist = Normal(pred["loc_mean"].view(-1, 2), pred["loc_sd"].view(-1, 2))
        locs_loss = -loc_dist.log_prob(true_tile_cat.locs.view(-1, 2)).sum(1)
        locs_loss *= true_tile_cat.is_on_array.view(-1)

        # star flux loss
        flux_dist = Normal(pred["log_flux_mean"].reshape(-1), pred["log_flux_sd"].view(-1))
        star_flux_loss = -flux_dist.log_prob(true_tile_cat["star_log_fluxes"].view(-1))
        star_flux_loss *= true_tile_cat.is_on_array.view(-1)
        star_flux_loss *= 1 - true_tile_cat["galaxy_bools"].view(-1)

        # star/galaxy classification
        galaxy_prob_flat = rearrange(pred["galaxy_prob"], "b ht wt 1 -> (b ht wt) 1")
        star_gal_prob = torch.cat([1 - galaxy_prob_flat, galaxy_prob_flat], dim=1)
        gal_bools_flat = true_tile_cat["galaxy_bools"].view(-1)
        binary_loss = -Categorical(star_gal_prob).log_prob(gal_bools_flat.long())
        binary_loss *= true_tile_cat.is_on_array.view(-1)

        loss = counter_loss + locs_loss + star_flux_loss + binary_loss

        return {
            "loss": loss.mean(),
            "counter_loss": counter_loss.mean(),
            "locs_loss": locs_loss.mean(),
            "star_flux_loss": star_flux_loss.mean(),
            "binary_loss": binary_loss.mean(),
        }

    def _generic_step(self, batch, logging_name, plot_images=False):
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
        self._generic_step(batch, "val", plot_images=True)
        return batch  # do we really need to save all these batches?

    def validation_epoch_end(self, outputs):
        """Pytorch lightning method."""
        batch: Dict[str, Tensor] = outputs[-1]
        self._generic_step(batch, "val")

    def test_step(self, batch, batch_idx):
        """Pytorch lightning method."""
        self._generic_step(batch, "test")
        return batch  # do we really need to save all these batches?
