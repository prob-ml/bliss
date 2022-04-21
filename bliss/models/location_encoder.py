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

    def __call__(self, image: Tensor, background: Tensor) -> Tensor:
        return torch.log1p(
            F.relu(image - background + self.z_threshold * background.sqrt(), inplace=True)
        )

    def output_channels(self, input_channels: int) -> int:
        return input_channels


class ConcatBackgroundTransform:
    def __init__(self):
        pass

    def __call__(self, image: Tensor, background: Tensor) -> Tensor:
        return torch.cat((image, background), dim=1)

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
        self.n_params_per_source = sum(param["dim"] for param in self.variational_params.values())

        # the number of total detections for all source counts: 1 + 2 + ... + self.max_detections
        # NOTE: the numerator here is always even
        n_total_detections = self.max_detections * (self.max_detections + 1) // 2

        # most of our parameters describe individual detections
        n_source_params = n_total_detections * self.n_params_per_source

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

        # get indices into the triangular array of returned parameters
        indx_mats = self._get_hidden_indices()
        for k, v in indx_mats.items():
            self.register_buffer(k + "_indx", v, persistent=False)

        # plotting
        self.annotate_probs = annotate_probs

        # metrics
        self.val_detection_metrics = DetectionMetrics(slack)
        self.test_detection_metrics = DetectionMetrics(slack)

    def encode(
        self, image: Tensor, background: Tensor, eval_mean_detections: Optional[float] = None
    ) -> Dict[str, Tensor]:
        """Encodes variational parameters from image padded tiles.

        Args:
            image: An astronomical image with shape `b * n_bands * h * w`.
            background: Background for `image` with the same shape as `image`.
            eval_mean_detections: If specified, adjusts the prior rate of object arrivals.

        Returns:
            A dictionary of two components:
            -  per_source_params:
                Tensor of shape b x n_tiles_h x n_tiles_w x D of variational parameters
                per tile.
            -  n_source_log_probs:
                Tensor of shape b x n_tiles_h x n_tiles_w x (max_sources + 1) indicating
                the log-probabilities of the number of sources present in each tile.
        """
        image2 = self.input_transform(image, background)
        image_ptiles = get_images_in_tiles(image2, self.tile_slen, self.ptile_slen)
        log_image_ptiles_flat: Tensor = rearrange(
            image_ptiles, "b nth ntw c h w -> (b nth ntw) c h w"
        )
        enc_conv_output = self.enc_conv(log_image_ptiles_flat)
        enc_final_output = self.enc_final(enc_conv_output)

        b = image_ptiles.shape[0]  # number of bands
        nth = image_ptiles.shape[1]  # number of horizontal tiles
        ntw = image_ptiles.shape[2]  # number of vertical tiles
        dim_out_all = enc_final_output.shape[1]
        dim_per_source_params = dim_out_all - (self.max_detections + 1)
        per_source_params_flat, n_source_free_probs_flat = torch.split(
            enc_final_output, (dim_per_source_params, self.max_detections + 1), dim=1
        )
        per_source_params = rearrange(
            per_source_params_flat, "(b nth ntw) d -> b nth ntw d", b=b, nth=nth, ntw=ntw
        )

        n_source_log_probs_flat = self.log_softmax(n_source_free_probs_flat)
        if eval_mean_detections is not None:
            train_log_probs = self._get_n_source_prior_log_prob(self.mean_detections)
            eval_log_probs = self._get_n_source_prior_log_prob(eval_mean_detections)
            adj = eval_log_probs - train_log_probs
            adj = rearrange(adj, "ns -> 1 ns")
            adj = adj.to(n_source_log_probs_flat.device)
            n_source_log_probs_flat += adj
        n_source_log_probs = rearrange(
            n_source_log_probs_flat, "(b nth ntw) ns -> b nth ntw ns", b=b, nth=nth, ntw=ntw
        )

        return {
            "per_source_params": per_source_params,
            "n_source_log_probs": n_source_log_probs,
        }

    def sample(self, var_params: Dict[str, Tensor], n_samples: int) -> Dict[str, Tensor]:
        """Sample from encoded variational distribution.

        Args:
            var_params: The output of `self.encode(ptiles)` which is the variational parameters
                in matrix form. Has size `n_ptiles * n_bands`.
            n_samples:
                The number of samples to draw

        Returns:
            A dictionary of tensors with shape `n_samples * n_ptiles * max_sources* ...`.
            Consists of `"n_sources", "locs", "log_fluxes", and "fluxes"`.
        """
        log_probs_n_sources_per_tile = var_params["n_source_log_probs"]
        # sample number of sources.
        # tile_n_sources shape = (n_samples x n_ptiles)
        # tile_is_on_array shape = (n_samples x n_ptiles x max_detections x 1)
        probs_n_sources_per_tile = torch.exp(log_probs_n_sources_per_tile)
        tile_n_sources = Categorical(probs=probs_n_sources_per_tile).sample((n_samples,))

        # get var_params conditioned on n_sources
        pred = self._encode_for_n_sources(
            rearrange(var_params["per_source_params"], "b nth ntw np -> (b nth ntw) np"),
            rearrange(tile_n_sources, "n b nth ntw -> n (b nth ntw)", n=n_samples),
        )

        tile_is_on_array = get_is_on_from_n_sources(tile_n_sources, self.max_detections)
        tile_is_on_array = tile_is_on_array.unsqueeze(-1).float()
        pred["loc_sd"] = torch.exp(0.5 * pred["loc_logvar"])
        pred["log_flux_sd"] = torch.exp(0.5 * pred["log_flux_logvar"])
        tile_locs = self._sample_gated_normal(
            pred["loc_mean"],
            pred["loc_sd"],
            rearrange(tile_is_on_array, "n b nth ntw ns 1 -> n (b nth ntw) ns 1"),
        )
        tile_log_fluxes = self._sample_gated_normal(
            pred["log_flux_mean"],
            pred["log_flux_sd"],
            rearrange(tile_is_on_array, "n b nth ntw ns 1 -> n (b nth ntw) ns 1"),
        )
        # Why are we masking here?
        tile_fluxes = tile_log_fluxes.exp() * rearrange(
            tile_is_on_array, "n b nth ntw ns 1 -> n (b nth ntw) ns 1"
        )
        sample_flat = {
            "locs": tile_locs,
            "log_fluxes": tile_log_fluxes,
            "fluxes": tile_fluxes,
        }
        b, nth, ntw, _ = log_probs_n_sources_per_tile.shape
        sample = {}
        for k, v in sample_flat.items():
            pattern = "ns (b nth ntw) s k -> ns b nth ntw s k"
            sample[k] = rearrange(v, pattern, b=b, nth=nth, ntw=ntw)
        sample["n_sources"] = tile_n_sources

        return sample

    def max_a_post(
        self, var_params: Dict[str, Tensor], n_source_weights: Optional[Tensor] = None
    ) -> TileCatalog:
        """Derive maximum a posteriori from variational parameters.

        Args:
            var_params: The output of `self.encode(ptiles)` which is the variational parameters
                in matrix form. Has size `n_ptiles * n_bands`.
            n_source_weights:
                If specified, adds adjustment to number of sources when taking the argmax. Useful
                for raising/lowering the threshold for turning sources on and off.

        Returns:
            The maximum a posteriori for each padded tile.
            Has shape `n_ptiles * max_detections * ...`.
            The dictionary contains
            `"locs", "log_fluxes", "fluxes", and "n_sources".`.
        """
        n_source_log_probs = var_params["n_source_log_probs"]
        if n_source_weights is None:
            n_source_weights = torch.ones(self.max_detections + 1)
        n_source_weights = n_source_weights.to(n_source_log_probs.device).reshape(1, 1, 1, -1)
        n_source_log_weights = n_source_weights.log()
        map_n_sources: Tensor = torch.argmax(n_source_log_probs + n_source_log_weights, dim=-1)

        pred = self._encode_for_n_sources(
            rearrange(var_params["per_source_params"], "b nth ntw np -> (b nth ntw) np"),
            rearrange(map_n_sources, "b nth ntw -> 1 (b nth ntw)"),
        )

        is_on_array = get_is_on_from_n_sources(map_n_sources, self.max_detections)
        is_on_array = is_on_array.unsqueeze(-1).float()

        # set sd so we return map estimates.
        # first locs
        locs_sd = torch.zeros_like(pred["loc_logvar"])
        tile_locs = self._sample_gated_normal(
            pred["loc_mean"],
            locs_sd,
            rearrange(is_on_array, "n nth ntw ns 1 -> (n nth ntw) ns 1"),
        )
        tile_locs = tile_locs.clamp(0, 1)

        # then log_fluxes
        log_flux_mean = pred["log_flux_mean"]
        log_flux_sd = torch.zeros_like(pred["log_flux_logvar"])
        tile_log_fluxes = self._sample_gated_normal(
            log_flux_mean,
            log_flux_sd,
            rearrange(is_on_array, "n nth ntw ns 1 -> (n nth ntw) ns 1"),
        )

        # Why are we masking here?
        tile_fluxes = tile_log_fluxes.exp() * rearrange(
            is_on_array, "n nth ntw ns 1 -> (n nth ntw) ns 1"
        )

        max_a_post_flat = {
            "locs": tile_locs,
            "log_fluxes": tile_log_fluxes,
            "fluxes": tile_fluxes,
        }
        max_a_post = {}
        b, nth, ntw, _ = var_params["per_source_params"].shape
        for k, v in max_a_post_flat.items():
            max_a_post[k] = rearrange(
                v, "1 (b nth ntw) s k -> b nth ntw s k", b=b, nth=nth, ntw=ntw
            )
        max_a_post.update(
            {
                "n_sources": map_n_sources,
                "n_source_log_probs": n_source_log_probs[:, :, :, 1:].unsqueeze(-1),
            }
        )
        return TileCatalog(self.tile_slen, max_a_post)

    def _get_n_source_prior_log_prob(self, detection_rate):
        possible_n_sources = torch.tensor(range(self.max_detections))
        log_probs = Poisson(torch.tensor(detection_rate)).log_prob(possible_n_sources)
        log_probs_last = torch.log1p(-torch.logsumexp(log_probs, 0).exp())
        return torch.cat((log_probs, log_probs_last.reshape(1)))

    def _encode_for_n_sources(
        self, var_params_flat: Tensor, tile_n_sources: Tensor
    ) -> Dict[str, Tensor]:
        """Get distributional parameters conditioned on number of sources in tile.

        Args:
            var_params_flat: The output of `self.encode(ptiles)`,
                where the first three dimensions have been flattened.
                These are the variational parameters in matrix form.
                Has size `(batch_size x n_tiles_h x n_tiles_w) * d`.
            tile_n_sources:
                A tensor of the number of sources in each tile.

        Returns:
            A dictionary where each member has shape
            `n_samples x n_ptiles x max_detections x ...`
        """
        tile_n_sources = tile_n_sources.clamp(max=self.max_detections)
        n_ptiles = var_params_flat.size(0)
        vpf_padded = F.pad(var_params_flat, (0, self.dim_out_all, 0, 0))

        var_params_for_n_sources = {}
        for k, param in self.variational_params.items():
            indx_mat = getattr(self, k + "_indx")
            indices = indx_mat[tile_n_sources.transpose(0, 1)].reshape(n_ptiles, -1)
            var_param = torch.gather(vpf_padded, 1, indices)

            var_params_for_n_sources[k] = rearrange(
                var_param,
                "np (ns d pd) -> ns np d pd",
                np=n_ptiles,
                ns=tile_n_sources.size(0),
                d=self.max_detections,
                pd=param["dim"],
            )

        # what?!? why is sigmoid(0) = 0?
        loc_mean_func = lambda x: torch.sigmoid(x) * (x != 0).float()
        # and why does location mean need to be transformed?
        # can't we stick with the original logistic-normal parameterization here?
        var_params_for_n_sources["loc_mean"] = loc_mean_func(var_params_for_n_sources["loc_mean"])

        return var_params_for_n_sources

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

        true_catalog = TileCatalog(
            self.tile_slen,
            {
                "locs": batch["locs"][:, :, :, 0 : self.max_detections],
                "log_fluxes": batch["log_fluxes"][:, :, :, 0 : self.max_detections],
                "galaxy_bools": batch["galaxy_bools"][:, :, :, 0 : self.max_detections],
                "n_sources": batch["n_sources"].clamp(max=self.max_detections),
            },
        )

        var_params = self.encode(batch["images"], batch["background"])
        n_source_log_probs = var_params["n_source_log_probs"]
        n_source_log_probs_flat = rearrange(n_source_log_probs, "n nth ntw ns -> (n nth ntw) ns")
        nll_loss = torch.nn.NLLLoss(reduction="none").requires_grad_(False)
        counter_loss = nll_loss(n_source_log_probs_flat, true_catalog.n_sources.reshape(-1))

        pred = self._encode_for_n_sources(
            rearrange(var_params["per_source_params"], "n nth ntw np -> (n nth ntw) np"),
            rearrange(true_catalog.n_sources, "n nth ntw -> 1 (n nth ntw)"),
        )
        locs_log_probs_all = _get_params_logprob_all_combs(
            rearrange(true_catalog.locs, "n nth ntw ns hw -> (n nth ntw) ns hw"),
            pred["loc_mean"].squeeze(0),
            pred["loc_logvar"].squeeze(0),
        )
        star_params_log_probs_all = _get_params_logprob_all_combs(
            rearrange(true_catalog["log_fluxes"], "n nth ntw ns nb -> (n nth ntw) ns nb"),
            pred["log_flux_mean"].squeeze(0),
            pred["log_flux_logvar"].squeeze(0),
        )

        (locs_loss, star_params_loss) = _get_min_perm_loss(
            locs_log_probs_all,
            star_params_log_probs_all,
            rearrange(true_catalog["galaxy_bools"], "n nth ntw ns 1 -> (n nth ntw) ns"),
            rearrange(true_catalog.is_on_array, "n nth ntw ns -> (n nth ntw) ns"),
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

        true_tile_catalog = TileCatalog(
            self.tile_slen,
            {
                "locs": batch["locs"][:, :, :, 0 : self.max_detections],
                "log_fluxes": batch["log_fluxes"][:, :, :, 0 : self.max_detections],
                "galaxy_bools": batch["galaxy_bools"][:, :, :, 0 : self.max_detections],
                "n_sources": batch["n_sources"].clamp(max=self.max_detections),
            },
        )
        true_full_catalog = true_tile_catalog.to_full_params()
        var_params = self.encode(batch["images"], batch["background"])
        est_tile_catalog = self.max_a_post(var_params)
        est_full_catalog = est_tile_catalog.to_full_params()

        metrics = self.val_detection_metrics(true_full_catalog, est_full_catalog)
        self.log("val/precision", metrics["precision"], batch_size=batch_size)
        self.log("val/recall", metrics["recall"], batch_size=batch_size)
        self.log("val/f1", metrics["f1"], batch_size=batch_size)
        self.log("val/avg_distance", metrics["avg_distance"], batch_size=batch_size)
        return batch

    def validation_epoch_end(self, outputs, kind="validation", max_n_samples=16):
        """Pytorch lightning method."""
        batch: Dict[str, Tensor] = outputs[-1]
        if self.n_bands > 1:
            return
        n_samples = min(int(math.sqrt(len(batch["n_sources"]))) ** 2, max_n_samples)
        nrows = int(n_samples**0.5)  # for figure

        true_tile_catalog = TileCatalog(
            self.tile_slen,
            {
                "locs": batch["locs"][:, :, :, 0 : self.max_detections],
                "log_fluxes": batch["log_fluxes"][:, :, :, 0 : self.max_detections],
                "galaxy_bools": batch["galaxy_bools"][:, :, :, 0 : self.max_detections],
                "star_bools": batch["star_bools"][:, :, :, 0 : self.max_detections],
                "n_sources": batch["n_sources"].clamp(max=self.max_detections),
            },
        )
        true_cat = true_tile_catalog.to_full_params()
        var_params = self.encode(batch["images"], batch["background"])
        est_tile_catalog = self.max_a_post(var_params)
        est_cat = est_tile_catalog.to_full_params()

        # setup figure and axes.
        fig, axes = plt.subplots(nrows=nrows, ncols=nrows, figsize=(12, 12))
        if nrows > 1:
            axes = axes.flatten()
        else:
            axes = [axes]

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
        true_tile_catalog = TileCatalog(
            self.tile_slen,
            {
                "locs": batch["locs"][:, :, :, 0 : self.max_detections],
                "log_fluxes": batch["log_fluxes"][:, :, :, 0 : self.max_detections],
                "galaxy_bools": batch["galaxy_bools"][:, :, :, 0 : self.max_detections],
                "n_sources": batch["n_sources"].clamp(max=self.max_detections),
            },
        )
        true_full_catalog = true_tile_catalog.to_full_params()
        var_params = self.encode(batch["images"], batch["background"])
        est_tile_catalog = self.max_a_post(var_params)
        est_full_catalog = est_tile_catalog.to_full_params()
        metrics = self.test_detection_metrics(true_full_catalog, est_full_catalog)
        batch_size = len(batch["images"])
        self.log("precision", metrics["precision"], batch_size=batch_size)
        self.log("recall", metrics["recall"], batch_size=batch_size)
        self.log("f1", metrics["f1"], batch_size=batch_size)
        self.log("avg_distance", metrics["avg_distance"], batch_size=batch_size)

        return batch

    @property
    def variational_params(self):
        return {
            "loc_mean": {"dim": 2},
            "loc_logvar": {"dim": 2},
            "log_flux_mean": {"dim": self.n_bands},
            "log_flux_logvar": {"dim": self.n_bands},
        }

    @staticmethod
    def _sample_gated_normal(mean, sd, tile_is_on_array):
        # tile_is_on_array can be either 'tile_is_on_array'/'tile_galaxy_bools'/'tile_star_bools'.
        # return shape = (n_samples x n_ptiles x max_detections x param_dim)
        assert tile_is_on_array.shape[-1] == 1
        return torch.normal(mean, sd) * tile_is_on_array

    def _get_hidden_indices(self):
        """Setup the indices corresponding to entries in h, cached since same for all h."""

        # initialize matrices containing the indices for each distributional param.
        indx_mats = {}
        for k, param in self.variational_params.items():
            param_dim = param["dim"]
            shape = (self.max_detections + 1, param_dim * self.max_detections)
            indx_mat = torch.full(shape, self.dim_out_all, dtype=torch.long)
            indx_mats[k] = indx_mat

        # add corresponding indices to the index matrices of distributional params
        # for a given n_detection.
        curr_indx = 0
        for n_detections in range(1, self.max_detections + 1):
            for k, param in self.variational_params.items():
                param_dim = param["dim"]
                new_indx = (param_dim * n_detections) + curr_indx
                indx_mats[k][n_detections, 0 : (param_dim * n_detections)] = torch.arange(
                    curr_indx, new_indx
                )
                curr_indx = new_indx

        return indx_mats


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


def _get_params_logprob_all_combs(true_params, param_mean, param_logvar):
    # return shape (n_ptiles x max_detections x max_detections)
    assert true_params.shape == param_mean.shape == param_logvar.shape

    n_ptiles = true_params.size(0)
    max_detections = true_params.size(1)

    # view to evaluate all combinations of log_prob.
    true_params = true_params.view(n_ptiles, 1, max_detections, -1)
    param_mean = param_mean.view(n_ptiles, max_detections, 1, -1)
    param_logvar = param_logvar.view(n_ptiles, max_detections, 1, -1)

    sd = (param_logvar.exp() + 1e-5).sqrt()
    return Normal(param_mean, sd).log_prob(true_params).sum(dim=3)
