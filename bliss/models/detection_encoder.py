import math
from typing import Dict, Optional, Union

import pytorch_lightning as pl
import torch
from einops import rearrange
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from omegaconf import OmegaConf
from torch import Tensor
from torch.distributions import Categorical, Normal
from torch.nn import functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
from yolov5.models.yolo import DetectionModel

from bliss.catalog import TileCatalog
from bliss.reporting import DetectionMetrics


class DetectionEncoder(pl.LightningModule):
    """Encodes the distribution of a latent variable representing an astronomical image.

    This class implements the source encoder, which is supposed to take in
    an astronomical image of size slen * slen and returns a NN latent variable
    representation of this image.
    """

    def __init__(
        self,
        architecture,
        n_bands: int,
        tile_slen: int,
        ptile_slen: int,
        annotate_probs: bool = False,
        slack: float = 1.0,
        optimizer_params: Optional[dict] = None,
        scheduler_params: Optional[dict] = None,
    ):
        """Initializes DetectionEncoder.

        Args:
            architecture: yaml to specifying the encoder network architecture
            n_bands: number of bands
            tile_slen: dimension of full image, we assume its square for now
            ptile_slen: dimension (in pixels) of the individual
                            image padded tiles (usually 8 for stars, and _ for galaxies).
            annotate_probs: Annotate probabilities on validation plots?
            slack: Slack to use when matching locations for validation metrics.
            optimizer_params: arguments passed to the Adam optimizer
            scheduler_params: arguments passed to the learning rate scheduler
        """
        super().__init__()

        self.n_bands = n_bands
        self.optimizer_params = optimizer_params
        self.scheduler_params = scheduler_params if scheduler_params else {"milestones": []}

        assert tile_slen <= ptile_slen
        self.tile_slen = tile_slen
        self.ptile_slen = ptile_slen

        assert (ptile_slen - tile_slen) % 2 == 0
        self.border_padding = (ptile_slen - tile_slen) // 2

        # Number of distributional parameters used to characterize each source in an image.
        self.n_params_per_source = sum(param["dim"] for param in self.dist_param_groups.values())

        # networks to be trained
        arch_dict = OmegaConf.to_container(architecture)
        self.model = DetectionModel(cfg=arch_dict, ch=2)
        self.tiles_to_crop = (ptile_slen - tile_slen) // (2 * tile_slen)

        # plotting
        self.annotate_probs = annotate_probs

        # metrics
        self.val_detection_metrics = DetectionMetrics(slack)
        self.test_detection_metrics = DetectionMetrics(slack)

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

    def _get_loss(self, batch: Dict[str, Tensor]):
        true_catalog = TileCatalog(self.tile_slen, batch)
        # next we clamp so that the the maximum number of true sources does not exceed the
        # support of the variational distribution
        true_catalog = true_catalog.truncate_sources(1)

        pred = self.encode_batch(batch)

        # counter loss
        on_prob_flat = rearrange(pred["on_prob"], "b ht wt 1 -> (b ht wt) 1")
        off_on_prob = torch.cat([1 - on_prob_flat, on_prob_flat], dim=1)
        true_on_flat = true_catalog.n_sources.reshape(-1)
        counter_loss = F.nll_loss(off_on_prob, true_on_flat, reduction="none")

        # location loss
        loc_dist = Normal(pred["loc_mean"].view(-1, 2), pred["loc_sd"].view(-1, 2))
        locs_loss = -(
            loc_dist.log_prob(true_catalog.locs.view(-1, 2)).sum(1)
            * true_catalog.is_on_array.view(-1)
        )

        # star flux loss
        flux_dist = Normal(pred["log_flux_mean"].reshape(-1), pred["log_flux_sd"].view(-1))
        star_params_loss = -(
            flux_dist.log_prob(true_catalog["star_log_fluxes"].view(-1))
            * true_catalog.is_on_array.view(-1)
        )

        # star/galaxy classification: calculate cross entropy loss only for "on" sources
        galaxy_prob_flat = rearrange(pred["galaxy_prob"], "b ht wt 1 -> (b ht wt) 1")
        star_gal_prob = torch.cat([1 - galaxy_prob_flat, galaxy_prob_flat], dim=1)
        gal_bools_flat = true_catalog["galaxy_bools"].view(-1)
        binary_loss = F.nll_loss(star_gal_prob, gal_bools_flat.long()) * true_on_flat

        # is this detach necessary?
        loss_vec = locs_loss * (locs_loss.detach() < 1e6).float()
        loss_vec += counter_loss + star_params_loss + binary_loss
        loss = loss_vec.mean()

        return {
            "loss": loss,
            "counter_loss": counter_loss,
            "locs_loss": locs_loss,
            "star_params_loss": star_params_loss,
            "binary_loss": binary_loss,
        }

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        """Training step (pytorch lightning)."""
        batch_size = len(batch["n_sources"])
        out = self._get_loss(batch)
        self.log("train/loss", out["loss"], batch_size=batch_size)
        return out["loss"]

    def _parse_batch(self, batch):
        pred = self.encode_batch(batch)

        true_tile_catalog = TileCatalog(self.tile_slen, batch)
        true_cat = true_tile_catalog.to_full_params()

        est_cat = self.variational_mode(pred)

        return true_cat, est_cat

    def validation_step(self, batch, batch_idx):
        """Pytorch lightning method."""
        batch_size = len(batch["images"])
        out = self._get_loss(batch)

        # log all losses
        self.log("val/loss", out["loss"], batch_size=batch_size)
        self.log("val/counter_loss", out["counter_loss"].mean(), batch_size=batch_size)
        self.log("val/locs_loss", out["locs_loss"].mean(), batch_size=batch_size)
        self.log("val/star_params_loss", out["star_params_loss"].mean(), batch_size=batch_size)

        true_cat, est_cat = self._parse_batch(batch)

        metrics = self.val_detection_metrics(true_cat, est_cat)
        self.log("val/precision", metrics["precision"], batch_size=batch_size)
        self.log("val/recall", metrics["recall"], batch_size=batch_size)
        self.log("val/f1", metrics["f1"], batch_size=batch_size)
        self.log("val/avg_distance", metrics["avg_distance"], batch_size=batch_size)
        return batch

    def plot_image_detections(self, images, true_cat, est_cat, nrows, img_ids):
        # setup figure and axes
        fig, axes = plt.subplots(nrows=nrows, ncols=nrows, figsize=(20, 20))
        axes = axes.flatten() if nrows > 1 else [axes]

        for ax_idx, ax in enumerate(axes):
            if ax_idx >= len(img_ids):
                break

            img_id = img_ids[ax_idx]
            true_n_sources = true_cat.n_sources[img_id].item()
            n_sources = est_cat.n_sources[img_id].item()
            ax.set_xlabel(f"True num: {true_n_sources}; Est num: {n_sources}")

            # add white border showing where centers of stars and galaxies can be
            bp = self.border_padding
            ax.axvline(bp, color="w")
            ax.axvline(images.shape[-1] - bp, color="w")
            ax.axhline(bp, color="w")
            ax.axhline(images.shape[-2] - bp, color="w")

            # plot image first
            image = images[img_id, 0].cpu().numpy()
            vmin = image.min().item()
            vmax = image.max().item()
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            im = ax.matshow(image, vmin=vmin, vmax=vmax, cmap="viridis")
            fig.colorbar(im, cax=cax, orientation="vertical")

            true_cat.plot_plocs(ax, img_id, "galaxy", bp=bp, color="r", marker="x", s=20)
            true_cat.plot_plocs(ax, img_id, "star", bp=bp, color="m", marker="x", s=20)
            est_cat.plot_plocs(ax, img_id, "all", bp=bp, color="b", marker="+", s=30)

            if ax_idx == 0:
                ax.scatter(None, None, color="r", marker="x", s=20, label="t.gal")
                ax.scatter(None, None, color="m", marker="x", s=20, label="t.star")
                ax.scatter(None, None, color="b", marker="+", s=30, label="p.source")
                ax.legend(
                    bbox_to_anchor=(0.0, 1.2, 1.0, 0.102),
                    loc="lower left",
                    ncol=2,
                    mode="expand",
                    borderaxespad=0.0,
                )

        fig.tight_layout()
        return fig

    def validation_epoch_end(self, outputs, kind="validation", max_n_samples=16):
        """Pytorch lightning method."""
        if self.n_bands > 1 or not self.logger:
            return

        batch: Dict[str, Tensor] = outputs[-1]
        true_cat, est_cat = self._parse_batch(batch)

        # log a grid of figures to the tensorboard
        batch_size = len(batch["images"])
        n_samples = min(int(math.sqrt(batch_size)) ** 2, max_n_samples)
        nrows = int(n_samples**0.5)  # for figure
        wrong_idx = (est_cat.n_sources != true_cat.n_sources).nonzero().view(-1)[:max_n_samples]
        fig = self.plot_image_detections(batch["images"], true_cat, est_cat, nrows, wrong_idx)
        title_root = f"Epoch:{self.current_epoch}/" if kind == "validation" else ""
        title = f"{title_root}{kind} images"
        self.logger.experiment.add_figure(title, fig)
        plt.close(fig)

    def test_step(self, batch, batch_idx):
        """Pytorch lightning method."""
        true_cat, est_cat = self._parse_batch(batch)

        batch_size = len(batch["images"])
        metrics = self.test_detection_metrics(true_cat, est_cat)
        self.log("precision", metrics["precision"], batch_size=batch_size)
        self.log("recall", metrics["recall"], batch_size=batch_size)
        self.log("f1", metrics["f1"], batch_size=batch_size)
        self.log("avg_distance", metrics["avg_distance"], batch_size=batch_size)

        return batch
