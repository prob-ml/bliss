# pylint: disable=R

from pathlib import Path
from typing import Optional

import numpy as np
import pytorch_lightning as pl
import torch
from einops import rearrange
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from torch import Tensor
from torch.distributions import LogNormal, Normal
from torch.optim import Adam

from bliss.catalog import TileCatalog, get_images_in_tiles
from bliss.models.encoder_layers import EncoderCNN, make_enc_final
from bliss.models.galaxy_encoder import CenterPaddedTilesTransform
from bliss.models.galsim_encoder import get_galsim_params_nll, sample_galsim_encoder


class LensEncoder(pl.LightningModule):
    def __init__(
        self,
        n_bands: int,
        tile_slen: int,
        ptile_slen: int,
        hidden: int,
        channel: int,
        spatial_dropout: float,
        dropout: float,
        optimizer_params: Optional[dict] = None,
        checkpoint_path: Optional[str] = None,
    ):
        super().__init__()

        self.n_bands = n_bands
        self.tile_slen = tile_slen
        self.ptile_slen = ptile_slen
        self.slen = self.ptile_slen - 2 * self.tile_slen  # will always crop 2 * tile_slen
        self.optimizer_params = optimizer_params

        assert (ptile_slen - tile_slen) % 2 == 0
        self.border_padding = (ptile_slen - tile_slen) // 2

        # will be trained.
        self.latent_dim = 12
        self.latent_dim_split = (
            7  # 7 for the lens galaxy parameters, 5 for pure lens (SIE) parameters
        )

        dim_enc_conv_out = ((self.slen + 1) // 2 + 1) // 2
        self.enc_conv = EncoderCNN(self.n_bands, channel, spatial_dropout)
        self.enc_final = make_enc_final(
            channel * 4 * dim_enc_conv_out**2,
            hidden,
            2 * self.latent_dim,
            dropout,
        )

        # grid for center cropped tiles
        self.center_ptiles = CenterPaddedTilesTransform(self.tile_slen, self.ptile_slen)

        # consistency
        assert self.slen >= 20, "Cropped slen is not reasonable for average sized galaxies."

        if checkpoint_path is not None:
            self.load_state_dict(
                torch.load(Path(checkpoint_path), map_location=torch.device("cpu"))
            )

    def configure_optimizers(self):
        """Set up optimizers (pytorch-lightning method)."""
        return Adam(self.parameters(), **self.optimizer_params)

    def forward(self, image_ptiles, tile_locs):
        raise NotImplementedError("Please use encode()")

    def encode(self, image_ptiles: Tensor, tile_locs: Tensor) -> Tensor:
        n_samples, n_ptiles, max_sources, _ = tile_locs.shape
        centered_ptiles = self._get_images_in_centered_tiles(image_ptiles, tile_locs)
        assert centered_ptiles.shape[-1] == centered_ptiles.shape[-2] == self.slen
        x = rearrange(centered_ptiles, "ns np c h w -> (ns np) c h w")
        enc_conv_output = self.enc_conv(x)
        lens_params_flat = self.enc_final(enc_conv_output)
        return rearrange(
            lens_params_flat,
            "(ns np ms) d -> ns np ms d",
            ns=n_samples,
            np=n_ptiles,
            ms=max_sources,
        )

    def sample(self, image_ptiles: Tensor, tile_locs: Tensor):
        var_dist_params = self.encode(image_ptiles, tile_locs)
        latent_dim_split = 2 * self.latent_dim_split
        lensed_galaxy_params, pure_lens_params = (
            var_dist_params[..., :latent_dim_split],
            var_dist_params[..., latent_dim_split:],
        )

        params_shape = list(var_dist_params[..., 0].shape) + [self.latent_dim]
        lens_params = torch.zeros(params_shape)
        lens_params[..., : self.latent_dim_split] = sample_galsim_encoder(lensed_galaxy_params)

        pure_lens_dim = self.latent_dim - self.latent_dim_split
        for param_idx in range(pure_lens_dim):
            dist_mean = pure_lens_params[..., 2 * param_idx]
            dist_logvar = pure_lens_params[..., 2 * param_idx + 1]
            dist_sd = (dist_logvar.exp() + 1e-5).sqrt()

            param = Normal(dist_mean, dist_sd).rsample()
            positive_param_idxs = {0}  # strictly positive are log-normal
            bounded_param_idxs = {3, 4}  # [0,1] bounded are logistic-normal

            if param_idx in positive_param_idxs:  # theta_E
                param = param.exp()
            elif param_idx in bounded_param_idxs:  # e1, e2
                param = torch.sigmoid(param) * 2 - 1
            lens_params[..., param_idx + self.latent_dim_split] = param

        return lens_params.to(device=var_dist_params.device)

    def training_step(self, batch, batch_idx):
        """Pytorch lightning training step."""
        batch_size = len(batch["images"])
        loss = self._get_loss(batch)
        self.log("train/loss", loss, batch_size=batch_size)
        return loss

    def validation_step(self, batch, batch_idx):
        """Pytorch lightning validation step."""
        batch_size = len(batch["images"])
        pred = self._get_loss(batch)
        self.log("val/loss", pred["loss"], batch_size=batch_size)
        pred_out = {f"pred_{k}": v for k, v in pred.items()}
        return {**batch, **pred_out}

    def validation_epoch_end(self, outputs):
        """Pytorch lightning method run at end of validation epoch."""
        # put all outputs together into a single batch
        batch = {}
        for b in outputs:
            for k, v in b.items():
                if v.shape:
                    curr_val = batch.get(k, torch.tensor([], device=v.device))
                    batch[k] = torch.cat([curr_val, v])
        if self.n_bands == 1:
            self._make_plots(batch)

    def _get_loss(self, batch):
        images: Tensor = batch["images"]
        background: Tensor = batch["background"]
        tile_catalog = TileCatalog(
            self.tile_slen, {k: v for k, v in batch.items() if k not in {"images", "background"}}
        )

        image_ptiles = get_images_in_tiles(
            torch.cat((images, background), dim=1),
            self.tile_slen,
            self.ptile_slen,
        )
        image_ptiles = rearrange(image_ptiles, "n nth ntw b h w -> (n nth ntw) b h w")
        locs = rearrange(tile_catalog.locs, "n nth ntw ns hw -> 1 (n nth ntw) ns hw")
        lens_pred = self.encode(image_ptiles, locs)
        lens_pred = rearrange(
            lens_pred,
            "ns (n nth ntw) ms d -> (ns n) nth ntw ms d",
            ns=1,
            nth=tile_catalog.n_tiles_h,
            ntw=tile_catalog.n_tiles_w,
        )

        lensed_galaxy_pred, pure_lens_pred = (
            lens_pred[..., : 2 * self.latent_dim_split],
            lens_pred[..., 2 * self.latent_dim_split :],
        )
        lensed_galaxy_params, pure_lens_params = (
            batch["lens_params"][..., : self.latent_dim_split],
            batch["lens_params"][..., self.latent_dim_split :],
        )

        loss_lens = self._get_params_logprob(
            batch["lensed_galaxy_bools"],
            pure_lens_params,
            pure_lens_pred,
        )

        loss_lensed_galaxy = get_galsim_params_nll(
            batch["lensed_galaxy_bools"],
            lensed_galaxy_params,
            lensed_galaxy_pred,
        )

        loss = loss_lens + loss_lensed_galaxy

        return {
            "loss": loss,
        }

    def _get_images_in_centered_tiles(self, image_ptiles: Tensor, tile_locs: Tensor) -> Tensor:
        n_bands = image_ptiles.shape[1] // 2
        img, bg = torch.split(image_ptiles, [n_bands, n_bands], dim=1)
        return self.center_ptiles(img - bg, tile_locs)

    def _get_params_logprob(self, lensed_galaxy_bools, true_params, params):
        # return shape (n_ptiles x max_detections x max_detections)
        assert true_params.shape[:-1] == params.shape[:-1]

        lensed_galaxy_bools = lensed_galaxy_bools.view(-1)
        true_params = true_params.view(-1, true_params.shape[-1]).transpose(0, 1)
        params = params.view(-1, params.shape[-1]).transpose(0, 1)

        # only compute loss where lensing is present
        true_params = true_params[:, lensed_galaxy_bools > 0]
        params = params[:, lensed_galaxy_bools > 0]

        # if no lenses present, skip loss calculation
        if true_params.shape[-1] == 0:
            return torch.zeros(1, requires_grad=True)

        theta_e, center_x, center_y, e1, e2 = true_params

        # all params are transformed to having support of (-∞, ∞) to allow modelling with normal
        # theta_E > 0, total_flux > 0, a_d > 0
        # e1, e2: [-1,1], beta_radians: [0, 2 * pi], disk_q: [0, 1]
        transformed_param_var_dist = [
            (theta_e, LogNormal),
            (center_x, Normal),
            (center_y, Normal),
            (torch.logit((e1 + 1) / 2), Normal),
            (torch.logit((e2 + 1) / 2), Normal),
        ]

        # compute log-likelihoods of parameters and negate at end for NLL loss
        log_prob = 0
        for i, (transformed_param, var_dist) in enumerate(transformed_param_var_dist):
            transformed_param_mean = params[2 * i]
            transformed_param_logvar = params[2 * i + 1]
            transformed_param_sd = (transformed_param_logvar.exp() + 1e-5).sqrt()

            parameterized_dist = var_dist(transformed_param_mean, transformed_param_sd)
            log_prob += parameterized_dist.log_prob(transformed_param).mean()

        return -log_prob

    # pylint: disable=too-many-statements
    def _make_plots(self, batch, n_samples=5):
        # validate worst reconstruction images.
        n_samples = min(len(batch["n_sources"]), n_samples)
        samples = np.random.choice(len(batch["n_sources"]), n_samples, replace=False)
        keys = [
            "images",
            "background",
            "locs",
            "galaxy_bools",
            "galaxy_params",
            "star_bools",
            "n_sources",
            "lensed_galaxy_bools",
            "lens_params",
        ]
        for k in keys:
            batch[k] = batch[k][samples]

        true_tile_catalog = TileCatalog(
            self.tile_slen,
            {
                "locs": batch["locs"],
                "galaxy_bools": batch["galaxy_bools"],
                "galaxy_params": batch["galaxy_params"],
                "star_bools": batch["star_bools"],
                "n_sources": batch["n_sources"],
                "lensed_galaxy_bools": batch["lensed_galaxy_bools"],
                "lens_params": batch["lens_params"],
            },
        )
        true_cat = true_tile_catalog.to_full_params()

        # extract non-params entries so that 'get_full_params' works.
        images = batch["images"]
        background = batch["background"]
        tile_locs = batch["locs"]

        # obtain map estimates
        image_ptiles = get_images_in_tiles(
            torch.cat((images, background), dim=1),
            self.tile_slen,
            self.ptile_slen,
        )
        _, n_tiles_h, n_tiles_w, _, _, _ = image_ptiles.shape
        image_ptiles = rearrange(image_ptiles, "n nth ntw b h w -> (n nth ntw) b h w")
        locs = rearrange(tile_locs, "n nth ntw ns hw -> 1 (n nth ntw) ns hw")
        z = self.encode(image_ptiles, locs)
        lens_params = rearrange(
            z,
            "ns (n nth ntw) ms d -> (ns n) nth ntw ms d",
            ns=1,
            nth=n_tiles_h,
            ntw=n_tiles_w,
        )

        tile_est = TileCatalog(
            self.tile_slen,
            {
                "locs": batch["locs"],
                "galaxy_bools": batch["galaxy_bools"],
                "galaxy_params": batch["galaxy_params"],
                "star_bools": batch["star_bools"],
                "n_sources": batch["n_sources"],
                "lensed_galaxy_bools": batch["lensed_galaxy_bools"],
                "lens_params": lens_params,
            },
        )
        est = tile_est.to_full_params()

        # setup figure and axes.
        nrows = int(n_samples**0.5)  # for figure
        fig, axes = plt.subplots(nrows=nrows, ncols=nrows, figsize=(12, 12))
        axes = axes.flatten() if nrows > 1 else [axes]

        assert images.shape[-2] == images.shape[-1]
        bp = self.border_padding
        for idx, ax in enumerate(axes):
            true_n_sources = true_cat.n_sources[idx].item()
            n_sources = est.n_sources[idx].item()
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
            est.plot_plocs(ax, idx, "all", bp=bp, color="b", marker="+", s=30)

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
            title = f"Epoch:{self.current_epoch}"
            self.logger.experiment.add_figure(title, fig)
        plt.close(fig)

    def test_step(self, batch, batch_idx):
        pass
