import math
from typing import Dict, Optional, Union

import pytorch_lightning as pl
import torch
from einops import rearrange
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from torch import Tensor
from torch.distributions import Categorical, Normal, Poisson
from torch.nn import functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR

from bliss.catalog import TileCatalog, get_images_in_tiles, get_is_on_from_n_sources
from bliss.models.encoder_layers import (
    ConcatBackgroundTransform,
    EncoderCNN,
    LogBackgroundTransform,
    make_enc_final,
)
from bliss.reporting import DetectionMetrics


class DetectionEncoder(pl.LightningModule):
    """Encodes the distribution of a latent variable representing an astronomical image.

    This class implements the source encoder, which is supposed to take in
    an astronomical image of size slen * slen and returns a NN latent variable
    representation of this image.
    """

    def __init__(
        self,
        input_transform: Union[LogBackgroundTransform, ConcatBackgroundTransform],
        max_detections: int,
        n_bands: int,
        tile_slen: int,
        ptile_slen: int,
        channel: int,
        dropout: float,
        hidden: int,
        spatial_dropout: float,
        annotate_probs: bool = False,
        slack: float = 1.0,
        optimizer_params: Optional[dict] = None,
        scheduler_params: Optional[dict] = None,
    ):
        """Initializes DetectionEncoder.

        Args:
            input_transform: Class which determines how input image and bg are transformed.
            max_detections: Number of maximum detections in a single tile.
            n_bands: number of bands
            tile_slen: dimension of full image, we assume its square for now
            ptile_slen: dimension (in pixels) of the individual
                            image padded tiles (usually 8 for stars, and _ for galaxies).
            channel: TODO (document this)
            spatial_dropout: TODO (document this)
            dropout: TODO (document this)
            hidden: TODO (document this)
            annotate_probs: Annotate probabilities on validation plots?
            slack: Slack to use when matching locations for validation metrics.
            optimizer_params: arguments passed to the Adam optimizer
            scheduler_params: arguments passed to the learning rate scheduler
        """
        super().__init__()

        self.input_transform = input_transform
        self.max_detections = max_detections
        assert max_detections == 1
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

        # the number of total detections for all source counts: 1 + 2 + ... + self.max_detections
        # NOTE: the numerator here is always even
        self.n_total_detections = self.max_detections * (self.max_detections + 1) // 2

        # most of our parameters describe individual detections
        n_source_params = self.n_total_detections * self.n_params_per_source

        # we also have parameters indicating the distribution of the number of detections
        count_simplex_dim = 1 + self.max_detections

        # the total number of distributional parameters per tile
        self.dim_out_all = n_source_params + count_simplex_dim

        dim_enc_conv_out = ((self.ptile_slen + 1) // 2 + 1) // 2

        # networks to be trained
        n_bands_in = self.input_transform.output_channels(n_bands)
        self.enc_conv = EncoderCNN(n_bands_in, channel, spatial_dropout)
        self.enc_final = make_enc_final(
            channel * 4 * dim_enc_conv_out**2,
            hidden,
            self.dim_out_all,
            dropout,
        )

        # the next block of code constructs `self.n_detections_map`, which is a 2d tensor with
        # size (self.max_detections + 1, self.max_detections).
        # There is one row for each possible number of detections (including zero).
        # Each row contains the indices of the relevant detections, padded by a dummy value.
        md, ntd = self.max_detections, self.n_total_detections
        n_detections_map = torch.full((md + 1, md), ntd, device=self.device)  # type: ignore
        tri = torch.tril_indices(md, md, device=self.device)  # type: ignore
        n_detections_map[tri[0] + 1, tri[1]] = torch.arange(ntd, device=self.device)  # type: ignore
        self.register_buffer("n_detections_map", n_detections_map)

        # plotting
        self.annotate_probs = annotate_probs

        # metrics
        self.val_detection_metrics = DetectionMetrics(slack)
        self.test_detection_metrics = DetectionMetrics(slack)

    @property
    def dist_param_groups(self):
        return {
            "loc_mean": {"dim": 2},
            "loc_logvar": {"dim": 2},
            "log_flux_mean": {"dim": self.n_bands},
            "log_flux_logvar": {"dim": self.n_bands},
        }

    def _final_encoding(self, enc_final_output):
        dim_out_all = enc_final_output.shape[1]
        dim_per_source_params = dim_out_all - (self.max_detections + 1)
        pred, n_source_free_probs = torch.split(
            enc_final_output, [dim_per_source_params, self.max_detections + 1], dim=1
        )
        per_source_params_tensor = rearrange(
            pred,
            "n_ptiles (td pps) -> n_ptiles td pps",
            td=self.n_total_detections,
            pps=self.n_params_per_source,
        )

        n_source_log_probs = F.log_softmax(n_source_free_probs, dim=1)

        split_sizes = [v["dim"] for v in self.dist_param_groups.values()]
        dist_params_split = torch.split(per_source_params_tensor.unsqueeze(0), split_sizes, 3)
        names = self.dist_param_groups.keys()
        pred = dict(zip(names, dist_params_split))

        pred["loc_mean"] = pred["loc_mean"].sigmoid()
        pred["loc_sd"] = (pred["loc_logvar"].exp() + 1e-5).sqrt()
        pred["log_flux_sd"] = (pred["log_flux_logvar"].exp() + 1e-5).sqrt()

        # delete these so we don't accidentally use them
        del pred["loc_logvar"]
        del pred["log_flux_logvar"]

        pred["n_source_log_prob"] = n_source_log_probs

        return pred

    def encode_tiled(self, image_ptiles: Tensor) -> Dict[str, Tensor]:
        """Encodes distributional parameters from image padded tiles.

        Args:
            image_ptiles: An astronomical image with shape `n_ptiles * n_bands * h * w`.

        Returns:
            A dictionary of two components:
            -  per_source_params:
                Tensor of shape b x n_tiles_h x n_tiles_w x D of distributional parameters
                per tile.
            -  n_source_log_probs:
                Tensor of shape b x n_tiles_h x n_tiles_w x (max_sources + 1) indicating
                the log-probabilities of the number of sources present in each tile.
        """
        transformed_ptiles = self.input_transform(image_ptiles)
        enc_conv_output = self.enc_conv(transformed_ptiles)
        enc_final_output = self.enc_final(enc_conv_output)
        return self._final_encoding(enc_final_output)

    def do_encode_batch(self, images_with_background):
        image_ptiles = get_images_in_tiles(
            images_with_background,
            self.tile_slen,
            self.ptile_slen,
        )
        image_ptiles = rearrange(image_ptiles, "n nth ntw b h w -> (n nth ntw) b h w")
        transformed_ptiles = self.input_transform(image_ptiles)
        enc_conv_output = self.enc_conv(transformed_ptiles)
        return self.enc_final(enc_conv_output)

    def encode_batch(self, batch):
        images_with_background = torch.cat((batch["images"], batch["background"]), dim=1)
        enc_final_output = self.do_encode_batch(images_with_background)
        return self._final_encoding(enc_final_output)

    def sample(
        self,
        pred: Dict[str, Tensor],
        n_samples: Union[int, None],
        n_source_weights: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """Sample from the encoded variational distribution.

        Args:
            pred:
                The distributional parameters in matrix form.
            n_samples:
                The number of samples to draw. If None, the variational mode is taken instead.
            n_source_weights:
                If specified, adjusts the sampling probabilities of n_sources.

        Returns:
            A dictionary of tensors with shape `n_samples * n_ptiles * max_sources * ...`.
            Consists of `"n_sources", "locs", "star_log_fluxes", and "star_fluxes"`.
        """
        if n_source_weights is None:
            max_n_weights = self.max_detections + 1
            n_source_weights = torch.ones(max_n_weights, device=self.device)  # type: ignore
        n_source_weights = n_source_weights.reshape(1, -1)
        ns_log_probs_adj = pred["n_source_log_prob"] + n_source_weights.log()
        ns_log_probs_adj -= ns_log_probs_adj.logsumexp(dim=-1, keepdim=True)

        if n_samples is not None:
            n_source_probs = ns_log_probs_adj.exp()
            tile_n_sources = Categorical(probs=n_source_probs).sample((n_samples,))
        else:
            tile_n_sources = torch.argmax(ns_log_probs_adj, dim=-1).unsqueeze(0)

        tile_is_on_array = get_is_on_from_n_sources(tile_n_sources, self.max_detections)
        tile_is_on_array = tile_is_on_array.unsqueeze(-1)

        if n_samples is not None:
            tile_locs = Normal(pred["loc_mean"], pred["loc_sd"]).rsample()
            tile_log_fluxes = Normal(pred["log_flux_mean"], pred["log_flux_sd"]).rsample()
        else:
            tile_locs = pred["loc_mean"]
            tile_log_fluxes = pred["log_flux_mean"]
        tile_fluxes = tile_log_fluxes.exp()
        tile_fluxes *= tile_is_on_array

        return {
            "locs": tile_locs,
            "star_log_fluxes": tile_log_fluxes,
            "star_fluxes": tile_fluxes,
            "n_sources": tile_n_sources,
        }

    def variational_mode(
        self, pred: Dict[str, Tensor], n_source_weights: Optional[Tensor] = None
    ) -> Dict[str, Tensor]:
        """Compute the variational mode. Special case of sample() where first dim is squeezed."""
        detection_params = self.sample(pred, None, n_source_weights=n_source_weights)
        return {k: v.squeeze(0) for k, v in detection_params.items()}

    def _get_n_source_prior_log_prob(self, detection_rate):
        possible_n_sources = torch.tensor(range(self.max_detections))
        log_probs = Poisson(torch.tensor(detection_rate)).log_prob(possible_n_sources)
        log_probs_last = torch.log1p(-torch.logsumexp(log_probs, 0).exp())
        return torch.cat((log_probs, log_probs_last.reshape(1)))

    # Pytorch Lightning methods

    def configure_optimizers(self):
        """Configure optimizers for training (pytorch lightning)."""
        optimizer = Adam(self.parameters(), **self.optimizer_params)
        scheduler = MultiStepLR(optimizer, **self.scheduler_params)
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        """Training step (pytorch lightning)."""
        batch_size = len(batch["n_sources"])
        out = self._get_loss(batch)
        self.log("train/loss", out["loss"], batch_size=batch_size)
        return out["loss"]

    def _get_loss(self, batch: Dict[str, Tensor]):
        true_catalog = TileCatalog(self.tile_slen, batch)
        # next we clamp so that the the maximum number of true sources does not exceed the
        # support of the variational distribution
        true_catalog = true_catalog.truncate_sources(self.max_detections)

        pred = self.encode_batch(batch)
        # no idea why we have these extra singleton dimensions. remove!
        assert pred["loc_mean"].size(0) == pred["loc_mean"].size(2) == 1

        truth_flat = true_catalog.n_sources.reshape(-1)
        counter_loss = F.nll_loss(pred["n_source_log_prob"], truth_flat, reduction="none")

        loc_dist = Normal(pred["loc_mean"][0, :, 0, :], pred["loc_sd"][0, :, 0, :])
        locs_loss = -(
            loc_dist.log_prob(true_catalog.locs.view(-1, 2)).sum(1)
            * true_catalog.is_on_array.view(-1)
        )

        flux_dist = Normal(pred["log_flux_mean"][0, :, 0, 0], pred["log_flux_sd"][0, :, 0, 0])
        star_params_loss = -(
            flux_dist.log_prob(true_catalog["star_log_fluxes"].view(-1))
            * true_catalog.is_on_array.view(-1)
        )

        loss_vec = locs_loss * (locs_loss.detach() < 1e6).float() + counter_loss + star_params_loss
        loss = loss_vec.mean()

        return {
            "loss": loss,
            "counter_loss": counter_loss,
            "locs_loss": locs_loss,
            "star_params_loss": star_params_loss,
        }

    def validation_step(self, batch, batch_idx):
        """Pytorch lightning method."""
        batch_size = len(batch["images"])
        out = self._get_loss(batch)

        # log all losses
        self.log("val/loss", out["loss"], batch_size=batch_size)
        self.log("val/counter_loss", out["counter_loss"].mean(), batch_size=batch_size)
        self.log("val/locs_loss", out["locs_loss"].mean(), batch_size=batch_size)
        self.log("val/star_params_loss", out["star_params_loss"].mean(), batch_size=batch_size)

        catalog_dict = {
            "locs": batch["locs"][:, :, :, 0 : self.max_detections],
            "star_log_fluxes": batch["star_log_fluxes"][:, :, :, 0 : self.max_detections],
            "galaxy_bools": batch["galaxy_bools"][:, :, :, 0 : self.max_detections],
            "n_sources": batch["n_sources"].clamp(max=self.max_detections),
        }
        true_tile_catalog = TileCatalog(self.tile_slen, catalog_dict)
        true_full_catalog = true_tile_catalog.to_full_params()

        pred = self.encode_batch(batch)

        est_catalog_dict = self.variational_mode(pred)
        est_tile_catalog = TileCatalog.from_flat_dict(
            true_tile_catalog.tile_slen,
            true_tile_catalog.n_tiles_h,
            true_tile_catalog.n_tiles_w,
            est_catalog_dict,
        )
        est_full_catalog = est_tile_catalog.to_full_params()

        metrics = self.val_detection_metrics(true_full_catalog, est_full_catalog)
        self.log("val/precision", metrics["precision"], batch_size=batch_size)
        self.log("val/recall", metrics["recall"], batch_size=batch_size)
        self.log("val/f1", metrics["f1"], batch_size=batch_size)
        self.log("val/avg_distance", metrics["avg_distance"], batch_size=batch_size)
        return batch

    def plot_image_detections(self, images, true_cat, est_cat, nrows, img_ids):
        # setup figure and axes.
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

    def _parse_batch(self, batch):
        pred = self.encode_batch(batch)

        true_tile_catalog = TileCatalog(self.tile_slen, batch)
        true_cat = true_tile_catalog.to_full_params()

        est_catalog_dict = self.variational_mode(pred)
        est_tile_catalog = TileCatalog.from_flat_dict(
            true_tile_catalog.tile_slen,
            true_tile_catalog.n_tiles_h,
            true_tile_catalog.n_tiles_w,
            est_catalog_dict,
        )
        est_cat = est_tile_catalog.to_full_params()

        return batch["images"], true_cat, est_cat

    def validation_epoch_end(self, outputs, kind="validation", max_n_samples=16):
        """Pytorch lightning method."""
        if self.n_bands > 1 or not self.logger:
            return

        batch: Dict[str, Tensor] = outputs[-1]
        images, true_cat, est_cat = self._parse_batch(batch)

        # log a grid of figures to the tensorboard
        batch_size = len(images)
        n_samples = min(int(math.sqrt(batch_size)) ** 2, max_n_samples)
        nrows = int(n_samples**0.5)  # for figure
        wrong_idx = (est_cat.n_sources != true_cat.n_sources).nonzero().view(-1)[:max_n_samples]
        fig = self.plot_image_detections(images, true_cat, est_cat, nrows, wrong_idx)
        title_root = f"Epoch:{self.current_epoch}/" if kind == "validation" else ""
        title = f"{title_root}{kind} images"
        self.logger.experiment.add_figure(title, fig)
        plt.close(fig)

    def test_step(self, batch, batch_idx):
        """Pytorch lightning method."""
        images, true_cat, est_cat = self._parse_batch(batch)

        batch_size = len(images)
        metrics = self.test_detection_metrics(true_cat, est_cat)
        self.log("precision", metrics["precision"], batch_size=batch_size)
        self.log("recall", metrics["recall"], batch_size=batch_size)
        self.log("f1", metrics["f1"], batch_size=batch_size)
        self.log("avg_distance", metrics["avg_distance"], batch_size=batch_size)

        return batch
