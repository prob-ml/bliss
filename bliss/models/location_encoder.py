import itertools
import math
from typing import Dict, Optional, Union

import pytorch_lightning as pl
import torch
from einops import rearrange
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from torch import Tensor, nn
from torch.distributions import Categorical, Normal, Poisson
from torch.nn import functional as F
from torch.optim import Adam

from bliss.catalog import TileCatalog, get_images_in_tiles, get_is_on_from_n_sources
from bliss.reporting import DetectionMetrics


class LogBackgroundTransform:
    def __init__(self, z_threshold: float = 4.0) -> None:
        self.z_threshold = z_threshold

    def __call__(self, image_and_background: Tensor) -> Tensor:
        n_bands = image_and_background.shape[1] // 2
        image, background = torch.split(image_and_background, (n_bands, n_bands), 1)
        return torch.log1p(
            F.relu(image - background + self.z_threshold * background.sqrt(), inplace=True)
        )

    def output_channels(self, input_channels: int) -> int:
        return input_channels


class ConcatBackgroundTransform:
    def __init__(self):
        pass

    def __call__(self, image_and_background: Tensor) -> Tensor:
        return image_and_background

    def output_channels(self, input_channels: int) -> int:
        return 2 * input_channels


class LocationEncoder(pl.LightningModule):
    """Encodes the distribution of a latent variable representing an astronomical image.

    This class implements the source encoder, which is supposed to take in
    an astronomical image of size slen * slen and returns a NN latent variable
    representation of this image.
    """

    def __init__(
        self,
        input_transform: Union[LogBackgroundTransform, ConcatBackgroundTransform],
        mean_detections: float,
        max_detections: int,
        n_bands: int,
        tile_slen: int,
        ptile_slen: int,
        channel: int,
        dropout,
        hidden: int,
        spatial_dropout: float,
        annotate_probs: bool = False,
        slack=1.0,
        optimizer_params: dict = None,
    ):
        """Initializes LocationEncoder.

        Args:
            input_transform: Class which determines how input image and bg are transformed.
            mean_detections: Poisson rate of detections in a single tile.
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
            optimizer_params: Optimizer for training.
        """
        super().__init__()

        self.input_transform = input_transform
        self.mean_detections = mean_detections
        self.max_detections = max_detections
        self.n_bands = n_bands
        self.optimizer_params = optimizer_params

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
        self.log_softmax = nn.LogSoftmax(dim=1)

        # the next block of code constructs `self.n_detections_map`, which is a 2d tensor with
        # size (self.max_detections + 1, self.max_detections).
        # There is one row for each possible number of detections (including zero).
        # Each row contains the indices of the relevant detections, padded by a dummy value.
        md, ntd = self.max_detections, self.n_total_detections
        n_detections_map = torch.full((md + 1, md), ntd, device=self.device)
        tri = torch.tril_indices(md, md, device=self.device)
        n_detections_map[tri[0] + 1, tri[1]] = torch.arange(ntd, device=self.device)
        self.register_buffer("n_detections_map", n_detections_map)

        # plotting
        self.annotate_probs = annotate_probs

        # metrics
        self.val_detection_metrics = DetectionMetrics(slack)
        self.test_detection_metrics = DetectionMetrics(slack)

    def encode(self, image_ptiles: Tensor) -> Dict[str, Tensor]:
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

        dim_out_all = enc_final_output.shape[1]
        dim_per_source_params = dim_out_all - (self.max_detections + 1)
        per_source_params, n_source_free_probs = torch.split(
            enc_final_output, (dim_per_source_params, self.max_detections + 1), dim=1
        )
        per_source_params = rearrange(
            per_source_params,
            "n_ptiles (td pps) -> n_ptiles td pps",
            td=self.n_total_detections,
            pps=self.n_params_per_source,
        )

        n_source_log_probs = self.log_softmax(n_source_free_probs)

        return {
            "per_source_params": per_source_params,
            "n_source_log_probs": n_source_log_probs,
        }

    def sample(self, dist_params: Dict[str, Tensor], n_samples: int) -> Dict[str, Tensor]:
        """Sample from the encoded variational distribution.

        Args:
            dist_params: The output of `self.encode(image_ptiles)`,
                which is the distributional parameters in matrix form.
            n_samples:
                The number of samples to draw

        Returns:
            A dictionary of tensors with shape `n_samples * n_ptiles * max_sources* ...`.
            Consists of `"n_sources", "locs", "log_fluxes", and "fluxes"`.
        """
        n_source_probs = dist_params["n_source_log_probs"].exp()
        tile_n_sources = Categorical(probs=n_source_probs).sample((n_samples,))

        # get distributional parameters conditioned on the sampled numbers of light sources
        dist_params_n_src = self._encode_for_n_sources(
            dist_params["per_source_params"], tile_n_sources
        )

        tile_is_on_array = get_is_on_from_n_sources(tile_n_sources, self.max_detections)
        tile_is_on_array = tile_is_on_array.unsqueeze(-1)

        tile_locs = Normal(dist_params_n_src["loc_mean"], dist_params_n_src["loc_sd"]).rsample()
        tile_locs *= tile_is_on_array  # Is masking here helpful/necessary?

        tile_log_fluxes = Normal(
            dist_params_n_src["log_flux_mean"], dist_params_n_src["log_flux_sd"]
        ).rsample()
        tile_fluxes = tile_log_fluxes.exp()
        tile_fluxes *= tile_is_on_array

        # Given all the rearranging/unflattening below, I suspect we never should have
        # flattened in the first place
        return {
            "locs": tile_locs,
            "log_fluxes": tile_log_fluxes,
            "fluxes": tile_fluxes,
            "n_sources": tile_n_sources,
        }

    def variational_mode(
        self, dist_params: Dict[str, Tensor], n_source_weights: Optional[Tensor] = None
    ) -> Dict[str, Tensor]:
        """Compute the mode of the variational distribution.

        Args:
            dist_params: The output of `self.encode(ptiles)` which is the distributional parameters
                in matrix form.
            n_source_weights:
                If specified, adds adjustment to number of sources when taking the argmax. Useful
                for raising/lowering the threshold for turning sources on and off.

        Returns:
            The maximum a posteriori for each padded tile.
            Has shape `n_ptiles * max_detections * ...`.
            The dictionary contains
            `"locs", "log_fluxes", "fluxes", and "n_sources".`.
        """
        if n_source_weights is None:
            n_source_weights = torch.ones(self.max_detections + 1, device=self.device)

        n_source_weights = n_source_weights.reshape(1, -1)
        log_joint = dist_params["n_source_log_probs"] + n_source_weights.log()
        map_n_sources: Tensor = torch.argmax(log_joint, dim=-1)

        # the first dimension is the number of samples (just one in this case)
        dist_params_n_src = self._encode_for_n_sources(
            dist_params["per_source_params"],
            map_n_sources.unsqueeze(0),
        )

        is_on_array = get_is_on_from_n_sources(map_n_sources, self.max_detections)
        is_on_array = rearrange(is_on_array, "n_ptiles ns -> n_ptiles ns 1")

        tile_locs = dist_params_n_src["loc_mean"] * is_on_array
        tile_locs = tile_locs.clamp(0, 1)

        # In the catalog, do we need to store both fluxes and log fluxes?
        # Also, can we avoid gating log_flux_mean here is_on_array? Because log(0) != 0.
        tile_log_fluxes = dist_params_n_src["log_flux_mean"] * is_on_array
        tile_fluxes = tile_log_fluxes.exp() * is_on_array

        # Given all the rearranging/unflattening below, perhaps we never should have
        # flattened in the first place
        unflatten = "1 n_ptiles s k -> n_ptiles s k"
        return {
            "locs": rearrange(tile_locs, unflatten),
            "log_fluxes": rearrange(tile_log_fluxes, unflatten),
            "fluxes": rearrange(tile_fluxes, unflatten),
            "n_sources": map_n_sources,
            "n_source_log_probs": dist_params["n_source_log_probs"][:, 1:].unsqueeze(-1),
        }

    def _get_n_source_prior_log_prob(self, detection_rate):
        possible_n_sources = torch.tensor(range(self.max_detections))
        log_probs = Poisson(torch.tensor(detection_rate)).log_prob(possible_n_sources)
        log_probs_last = torch.log1p(-torch.logsumexp(log_probs, 0).exp())
        return torch.cat((log_probs, log_probs_last.reshape(1)))

    def _encode_for_n_sources(
        self, params_per_source: Tensor, tile_n_sources: Tensor
    ) -> Dict[str, Tensor]:
        """Get distributional parameters conditioned on number of sources in tile.

        Args:
            params_per_source: An output of `self.encode(ptiles)`,
                Has size `(batch_size x n_tiles_h x n_tiles_w) * d`.
            tile_n_sources:
                A tensor of the number of sources in each tile.

        Returns:
            A dictionary where each member has shape
            `n_samples x n_ptiles x max_detections x ...`
        """
        assert tile_n_sources.max() <= self.max_detections

        # first, we transform `tile_n_sources` so that it can be used as an index
        # for looking up detections in `params_per_source`
        sindx1 = self.n_detections_map[tile_n_sources]  # type: ignore
        sindx2 = rearrange(sindx1, "ns np md -> np (ns md) 1")
        sindx3 = sindx2.expand(sindx2.size(0), sindx2.size(1), self.n_params_per_source)

        # next, we pad `params_per_source` with a dummy column of zeros that will be looked up
        # (copied) whenever fewer the `max_detections` sources are present. `gather` does the copy.
        pps_padded = F.pad(params_per_source, (0, 0, 0, 1))
        pps_gathered = torch.gather(pps_padded, 1, sindx3)
        params_n_srcs_combined = rearrange(
            pps_gathered, "np (ns md) pps -> ns np md pps", ns=tile_n_sources.size(0)
        )

        # finally, we slice pps5 by parameter group because these groups are treated differently,
        # subsequently
        split_sizes = [v["dim"] for v in self.dist_param_groups.values()]
        dist_params_split = torch.split(params_n_srcs_combined, split_sizes, 3)
        names = self.dist_param_groups.keys()
        params_n_srcs = dict(zip(names, dist_params_split))

        # I don't think the special case for `x == 0` should be necessary
        loc_mean_func = lambda x: torch.sigmoid(x) * (x != 0).float()
        params_n_srcs["loc_mean"] = loc_mean_func(params_n_srcs["loc_mean"])

        params_n_srcs["loc_sd"] = (params_n_srcs["loc_logvar"].exp() + 1e-5).sqrt()
        params_n_srcs["log_flux_sd"] = (params_n_srcs["log_flux_logvar"].exp() + 1e-5).sqrt()

        # delete these so we don't accidentally use them
        del params_n_srcs["loc_logvar"]
        del params_n_srcs["log_flux_logvar"]

        return params_n_srcs

    # Pytorch Lightning methods

    def configure_optimizers(self):
        """Configure optimizers for training (pytorch lightning)."""
        return Adam(self.parameters(), **self.optimizer_params)

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        """Training step (pytorch lightning)."""
        batch_size = len(batch["n_sources"])
        loss = self._get_loss(batch)["loss"]
        self.log("train/loss", loss, batch_size=batch_size)
        return loss

    def _get_loss(self, batch: Dict[str, Tensor]):
        true_catalog = {
            "locs": rearrange(
                batch["locs"][:, :, :, 0 : self.max_detections],
                "n nth ntw ns hw -> (n nth ntw) ns hw",
            ),
            "log_fluxes": rearrange(
                batch["log_fluxes"][:, :, :, 0 : self.max_detections],
                "n nth ntw ns b -> (n nth ntw) ns b",
            ),
            "galaxy_bools": rearrange(
                batch["galaxy_bools"][:, :, :, 0 : self.max_detections],
                "n nth ntw ns 1 -> (n nth ntw) ns 1",
            ),
            "n_sources": rearrange(
                batch["n_sources"].clamp(max=self.max_detections), "n nth ntw -> (n nth ntw)"
            ),
        }
        true_catalog["is_on_array"] = get_is_on_from_n_sources(
            true_catalog["n_sources"], self.max_detections
        )
        image_ptiles = get_images_in_tiles(
            torch.cat((batch["images"], batch["background"]), dim=1),
            self.tile_slen,
            self.ptile_slen,
        )
        image_ptiles = rearrange(image_ptiles, "n nth ntw b h w -> (n nth ntw) b h w")
        dist_params = self.encode(image_ptiles)
        nslp_flat = rearrange(dist_params["n_source_log_probs"], "n_ptiles ns -> n_ptiles ns")
        counter_loss = F.nll_loss(
            nslp_flat, true_catalog["n_sources"].reshape(-1), reduction="none"
        )

        pred = self._encode_for_n_sources(
            dist_params["per_source_params"],
            rearrange(true_catalog["n_sources"], "n_ptiles -> 1 n_ptiles"),
        )
        locs_log_probs_all = _get_params_logprob_all_combs(
            true_catalog["locs"],
            pred["loc_mean"].squeeze(0),
            pred["loc_sd"].squeeze(0),
        )
        star_params_log_probs_all = _get_params_logprob_all_combs(
            true_catalog["log_fluxes"],
            pred["log_flux_mean"].squeeze(0),
            pred["log_flux_sd"].squeeze(0),
        )

        (locs_loss, star_params_loss) = _get_min_perm_loss(
            locs_log_probs_all,
            star_params_log_probs_all,
            rearrange(true_catalog["galaxy_bools"], "n_ptiles ns 1 -> n_ptiles ns"),
            true_catalog["is_on_array"],
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
            "log_fluxes": batch["log_fluxes"][:, :, :, 0 : self.max_detections],
            "galaxy_bools": batch["galaxy_bools"][:, :, :, 0 : self.max_detections],
            "n_sources": batch["n_sources"].clamp(max=self.max_detections),
        }
        true_tile_catalog = TileCatalog(self.tile_slen, catalog_dict)
        true_full_catalog = true_tile_catalog.to_full_params()
        image_ptiles = get_images_in_tiles(
            torch.cat((batch["images"], batch["background"]), dim=1),
            self.tile_slen,
            self.ptile_slen,
        )
        image_ptiles = rearrange(image_ptiles, "n nth ntw b h w -> (n nth ntw) b h w")
        dist_params = self.encode(image_ptiles)
        est_catalog_dict = self.variational_mode(dist_params)
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

    def validation_epoch_end(self, outputs, kind="validation", max_n_samples=16):
        # pylint: disable=too-many-statements
        """Pytorch lightning method."""
        batch: Dict[str, Tensor] = outputs[-1]
        if self.n_bands > 1:
            return
        n_samples = min(int(math.sqrt(len(batch["n_sources"]))) ** 2, max_n_samples)
        nrows = int(n_samples**0.5)  # for figure

        catalog_dict = {
            "locs": batch["locs"][:, :, :, 0 : self.max_detections],
            "log_fluxes": batch["log_fluxes"][:, :, :, 0 : self.max_detections],
            "galaxy_bools": batch["galaxy_bools"][:, :, :, 0 : self.max_detections],
            "star_bools": batch["star_bools"][:, :, :, 0 : self.max_detections],
            "n_sources": batch["n_sources"].clamp(max=self.max_detections),
        }
        true_tile_catalog = TileCatalog(self.tile_slen, catalog_dict)
        true_cat = true_tile_catalog.to_full_params()

        image_ptiles = get_images_in_tiles(
            torch.cat((batch["images"], batch["background"]), dim=1),
            self.tile_slen,
            self.ptile_slen,
        )
        image_ptiles = rearrange(image_ptiles, "n nth ntw b h w -> (n nth ntw) b h w")
        dist_params = self.encode(image_ptiles)

        est_catalog_dict = self.variational_mode(dist_params)
        est_tile_catalog = TileCatalog.from_flat_dict(
            true_tile_catalog.tile_slen,
            true_tile_catalog.n_tiles_h,
            true_tile_catalog.n_tiles_w,
            est_catalog_dict,
        )
        est_cat = est_tile_catalog.to_full_params()

        # setup figure and axes.
        fig, axes = plt.subplots(nrows=nrows, ncols=nrows, figsize=(12, 12))
        axes = axes.flatten() if nrows > 1 else [axes]

        images = batch["images"]
        assert images.shape[-2] == images.shape[-1]
        bp = self.border_padding
        for idx, ax in enumerate(axes):
            true_n_sources = true_cat.n_sources[idx].item()
            n_sources = est_cat.n_sources[idx].item()
            ax.set_xlabel(f"True num: {true_n_sources}; Est num: {n_sources}")

            # add white border showing where centers of stars and galaxies can be
            ax.axvline(bp, color="w")
            ax.axvline(images.shape[-1] - bp, color="w")
            ax.axhline(bp, color="w")
            ax.axhline(images.shape[-2] - bp, color="w")

            # plot image first
            image = images[idx, 0].cpu().numpy()
            vmin = image.min().item()
            vmax = image.max().item()
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            im = ax.matshow(image, vmin=vmin, vmax=vmax, cmap="viridis")
            fig.colorbar(im, cax=cax, orientation="vertical")

            true_cat.plot_plocs(ax, idx, "galaxy", bp=bp, color="r", marker="x", s=20)
            true_cat.plot_plocs(ax, idx, "star", bp=bp, color="c", marker="x", s=20)
            est_cat.plot_plocs(ax, idx, "all", bp=bp, color="b", marker="+", s=30)

            if idx == 0:
                ax.scatter(None, None, color="r", marker="x", s=20, label="t.gal")
                ax.scatter(None, None, color="c", marker="x", s=20, label="t.star")
                ax.scatter(None, None, color="b", marker="+", s=30, label="p.source")
                ax.legend(
                    bbox_to_anchor=(0.0, 1.2, 1.0, 0.102),
                    loc="lower left",
                    ncol=2,
                    mode="expand",
                    borderaxespad=0.0,
                )

        fig.tight_layout()
        if self.logger:
            if kind == "validation":
                title = f"Epoch:{self.current_epoch}/Validation Images"
                self.logger.experiment.add_figure(title, fig)
            elif kind == "testing":
                self.logger.experiment.add_figure("Test Images", fig)
            else:
                raise NotImplementedError()
        plt.close(fig)

    def test_step(self, batch, batch_idx):
        """Pytorch lightning method."""
        catalog_dict = {
            "locs": batch["locs"][:, :, :, 0 : self.max_detections],
            "log_fluxes": batch["log_fluxes"][:, :, :, 0 : self.max_detections],
            "galaxy_bools": batch["galaxy_bools"][:, :, :, 0 : self.max_detections],
            "n_sources": batch["n_sources"].clamp(max=self.max_detections),
        }
        true_tile_catalog = TileCatalog(self.tile_slen, catalog_dict)
        true_full_catalog = true_tile_catalog.to_full_params()

        image_ptiles = get_images_in_tiles(
            torch.cat((batch["images"], batch["background"]), dim=1),
            self.tile_slen,
            self.ptile_slen,
        )
        image_ptiles = rearrange(image_ptiles, "n nth ntw b h w -> (n nth ntw) b h w")
        dist_params = self.encode(image_ptiles)

        est_catalog_dict = self.variational_mode(dist_params)
        est_tile_catalog = TileCatalog.from_flat_dict(
            self.tile_slen,
            true_tile_catalog.n_tiles_h,
            true_tile_catalog.n_tiles_w,
            est_catalog_dict,
        )
        est_full_catalog = est_tile_catalog.to_full_params()
        metrics = self.test_detection_metrics(true_full_catalog, est_full_catalog)
        batch_size = len(batch["images"])
        self.log("precision", metrics["precision"], batch_size=batch_size)
        self.log("recall", metrics["recall"], batch_size=batch_size)
        self.log("f1", metrics["f1"], batch_size=batch_size)
        self.log("avg_distance", metrics["avg_distance"], batch_size=batch_size)

        return batch

    @property
    def dist_param_groups(self):
        return {
            "loc_mean": {"dim": 2},
            "loc_logvar": {"dim": 2},
            "log_flux_mean": {"dim": self.n_bands},
            "log_flux_logvar": {"dim": self.n_bands},
        }


def make_enc_final(in_size, hidden, out_size, dropout):
    return nn.Sequential(
        nn.Flatten(1),
        nn.Linear(in_size, hidden),
        nn.BatchNorm1d(hidden),
        nn.ReLU(True),
        nn.Dropout(dropout),
        nn.Linear(hidden, hidden),
        nn.BatchNorm1d(hidden),
        nn.ReLU(True),
        nn.Dropout(dropout),
        nn.Linear(hidden, out_size),
    )


class EncoderCNN(nn.Module):
    def __init__(self, n_bands, channel, dropout):
        super().__init__()
        self.layer = self._make_layer(n_bands, channel, dropout)

    def forward(self, x: Tensor) -> Tensor:
        """Runs encoder CNN on inputs."""
        return self.layer(x)

    def _make_layer(self, n_bands, channel, dropout):
        layers = [
            nn.Conv2d(n_bands, channel, 3, padding=1),
            nn.BatchNorm2d(channel),
            nn.ReLU(True),
        ]
        in_channel = channel
        for i in range(3):
            downsample = i != 0
            layers += [
                ConvBlock(in_channel, channel, dropout, downsample),
                ConvBlock(channel, channel, dropout),
                ConvBlock(channel, channel, dropout),
            ]
            in_channel = channel
            channel = channel * 2
        return nn.Sequential(*layers)


class ConvBlock(nn.Module):
    """A Convolution Layer.

    This module is two stacks of Conv2D -> ReLU -> BatchNorm, with dropout
    in the middle, and an option to downsample with a stride of 2.

    Parameters:
        in_channel: Number of input channels
        out_channel: Number of output channels
        dropout: Dropout proportion between [0, 1]
        downsample (optional): Whether to downsample with stride of 2.
    """

    def __init__(self, in_channel: int, out_channel: int, dropout: float, downsample: bool = False):
        """Initializes the module layers."""
        super().__init__()
        self.downsample = downsample
        stride = 1
        if self.downsample:
            stride = 2
            self.sc_conv = nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=stride)
            self.sc_bn = nn.BatchNorm2d(out_channel)
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1, stride=stride)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.drop1 = nn.Dropout2d(dropout)
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channel)

    def forward(self, x: Tensor) -> Tensor:
        """Runs convolutional block on inputs."""
        identity = x

        x = self.conv1(x)
        x = F.relu(self.bn1(x), inplace=True)

        x = self.drop1(x)

        x = self.conv2(x)
        x = self.bn2(x)

        if self.downsample:
            identity = self.sc_bn(self.sc_conv(identity))

        out = x + identity
        return F.relu(out, inplace=True)


def _get_log_probs_all_perms(
    locs_log_probs_all,
    star_params_log_probs_all,
    true_galaxy_bools,
    is_on_array,
):
    # get log-probability under every possible matching of estimated source to true source
    n_ptiles = star_params_log_probs_all.size(0)
    max_detections = star_params_log_probs_all.size(-1)

    n_permutations = math.factorial(max_detections)
    locs_log_probs_all_perm = torch.zeros(
        n_ptiles, n_permutations, device=locs_log_probs_all.device
    )
    star_params_log_probs_all_perm = locs_log_probs_all_perm.clone()

    for i, perm in enumerate(itertools.permutations(range(max_detections))):
        # note that we multiply is_on_array, we only evaluate the loss if the source is on.
        locs_log_probs_all_perm[:, i] = (
            locs_log_probs_all[:, perm].diagonal(dim1=1, dim2=2) * is_on_array
        ).sum(1)

        # if star, evaluate the star parameters,
        # hence the multiplication by (1 - true_galaxy_bools)
        # the diagonal is a clever way of selecting the elements of each permutation (first index
        # of mean/var with second index of true_param etc.)
        star_params_log_probs_all_perm[:, i] = (
            star_params_log_probs_all[:, perm].diagonal(dim1=1, dim2=2)
            * is_on_array
            * (1 - true_galaxy_bools)
        ).sum(1)

    return locs_log_probs_all_perm, star_params_log_probs_all_perm


def _get_min_perm_loss(
    locs_log_probs_all,
    star_params_log_probs_all,
    true_galaxy_bools,
    is_on_array,
):
    # get log-probability under every possible matching of estimated star to true star
    locs_log_probs_all_perm, star_params_log_probs_all_perm = _get_log_probs_all_perms(
        locs_log_probs_all,
        star_params_log_probs_all,
        true_galaxy_bools,
        is_on_array,
    )

    # find the permutation that minimizes the location losses
    locs_loss, indx = torch.min(-locs_log_probs_all_perm, dim=1)
    indx = indx.unsqueeze(1)

    # get the star losses according to the found permutation.
    star_params_loss = -torch.gather(star_params_log_probs_all_perm, 1, indx).squeeze()
    return locs_loss, star_params_loss


def _get_params_logprob_all_combs(true_params, param_mean, param_sd):
    # return shape (n_ptiles x max_detections x max_detections)
    assert true_params.shape == param_mean.shape == param_sd.shape

    n_ptiles = true_params.size(0)
    max_detections = true_params.size(1)

    # view to evaluate all combinations of log_prob.
    true_params = true_params.view(n_ptiles, 1, max_detections, -1)
    param_mean = param_mean.view(n_ptiles, max_detections, 1, -1)
    param_sd = param_sd.view(n_ptiles, max_detections, 1, -1)

    return Normal(param_mean, param_sd).log_prob(true_params).sum(dim=3)
