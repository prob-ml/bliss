from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import pytorch_lightning as pl
import torch
from einops import rearrange
from matplotlib import pyplot as plt
from torch import Tensor, nn
from torch.distributions import Normal
from torch.nn import functional as F
from torch.optim import Adam

from bliss.catalog import TileCatalog, get_images_in_tiles
from bliss.models.decoder import ImageDecoder, get_mgrid
from bliss.models.galaxy_net import CenteredGalaxyEncoder, OneCenteredGalaxyAE
from bliss.models.vae.galaxy_flow import CenteredGalaxyLatentFlow
from bliss.models.vae.galaxy_vae import OneCenteredGalaxyVAE
from bliss.plotting import add_loc_legend, plot_image, plot_locs


class GalaxyEncoder(pl.LightningModule):
    def __init__(
        self,
        decoder: ImageDecoder,
        autoencoder: Union[OneCenteredGalaxyAE, OneCenteredGalaxyVAE],
        hidden: int,
        vae_flow: Optional[CenteredGalaxyLatentFlow] = None,
        vae_flow_ckpt: Optional[str] = None,
        optimizer_params: Optional[dict] = None,
        crop_loss_at_border: bool = False,
        checkpoint_path: Optional[str] = None,
        max_flux_valid_plots: Optional[int] = None,
    ):
        super().__init__()

        self.crop_loss_at_border = crop_loss_at_border
        self.max_flux_valid_plots = max_flux_valid_plots
        self.optimizer_params = optimizer_params

        # to produce images to train on.
        self.image_decoder = decoder
        self.image_decoder.requires_grad_(False)

        # extract useful info from image_decoder
        self.n_bands = self.image_decoder.n_bands

        # put image dimensions together
        self.tile_slen = self.image_decoder.tile_slen
        self.border_padding = self.image_decoder.border_padding
        self.ptile_slen = self.tile_slen + 2 * self.border_padding
        self.slen = self.ptile_slen - 2 * self.tile_slen  # will always crop 2 * tile_slen

        # will be trained.
        self.enc = autoencoder.make_deblender(
            self.slen, autoencoder.latent_dim, self.n_bands, hidden
        )
        if vae_flow is not None:
            if vae_flow_ckpt is None:
                raise TypeError("Using `vae_flow` requires `vae_flow_ckpt`.")
            state_dict = torch.load(vae_flow_ckpt, map_location=vae_flow.device)  # type: ignore
            vae_flow.load_state_dict(state_dict)
            vae_flow.eval()
            vae_flow.requires_grad_(False)
            self.enc.p_z = vae_flow
        self.latent_dim = autoencoder.latent_dim

        # grid for center cropped tiles
        self.center_ptiles = CenterPaddedTilesTransform(self.tile_slen, self.ptile_slen)

        # consistency
        assert self.slen >= 20, "Cropped slen is not reasonable for average sized galaxies."

        if checkpoint_path is not None:
            self.load_state_dict(
                torch.load(Path(checkpoint_path), map_location=torch.device("cpu"))
            )

    def sample(
        self, image_ptiles: Tensor, tile_locs: Tensor, deterministic: Optional[bool]
    ) -> Tensor:
        n_samples, n_ptiles, max_sources, _ = tile_locs.shape
        centered_ptiles = self._get_images_in_centered_tiles(image_ptiles, tile_locs)
        assert centered_ptiles.shape[-1] == centered_ptiles.shape[-2] == self.slen
        x = rearrange(centered_ptiles, "ns np c h w -> (ns np) c h w")
        if deterministic is None:
            deterministic = isinstance(self.enc, CenteredGalaxyEncoder)
        galaxy_params_flat = self.enc.sample(x, deterministic=deterministic)
        return rearrange(
            galaxy_params_flat,
            "(ns np ms) d -> ns np ms d",
            ns=n_samples,
            np=n_ptiles,
            ms=max_sources,
        )

    def training_step(self, batch, batch_idx):
        """Pytorch lightning training step."""
        batch_size = len(batch["images"])
        loss = self._get_loss(batch)
        self.log("train/loss", loss, batch_size=batch_size)
        return loss

    def validation_step(self, batch, batch_idx):
        """Pytorch lightning validation step."""
        batch_size = len(batch["images"])
        loss = self._get_loss(batch)
        self.log("val/loss", loss, batch_size=batch_size)
        return batch

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
        galaxy_params, pq_divergence = self._encode(image_ptiles, locs)
        # draw fully reconstructed image.
        # NOTE: Assume recon_mean = recon_var per poisson approximation.
        tile_catalog["galaxy_params"] = rearrange(
            galaxy_params,
            "ns (n nth ntw) ms d -> (ns n) nth ntw ms d",
            nth=tile_catalog.n_tiles_h,
            ntw=tile_catalog.n_tiles_w,
        )
        recon_mean = self.image_decoder.render_images(tile_catalog)
        recon_mean = rearrange(recon_mean, "(ns n) c h w -> ns n c h w", ns=1)
        recon_mean += background.unsqueeze(0)

        assert not torch.any(torch.isnan(recon_mean))
        assert not torch.any(torch.isinf(recon_mean))
        recon_losses = -Normal(recon_mean, recon_mean.sqrt()).log_prob(images.unsqueeze(0))
        if self.crop_loss_at_border:
            bp = self.border_padding * 2
            recon_losses = recon_losses[:, :, :, bp:(-bp), bp:(-bp)]
        assert not torch.any(torch.isnan(recon_losses))
        assert not torch.any(torch.isinf(recon_losses))

        # For divergence loss, we only evaluate tiles with a galaxy in them
        galaxy_bools = rearrange(tile_catalog["galaxy_bools"], "n nth ntw ms 1 -> 1 (n nth ntw) ms")
        divergence_loss = (pq_divergence * galaxy_bools).sum()
        return recon_losses.sum() - divergence_loss

    def _encode(self, image_ptiles: Tensor, tile_locs: Tensor) -> Tuple[Tensor, Tensor]:
        """Runs galaxy encoder on input image ptiles (with bg substracted)."""
        n_samples, n_ptiles, max_sources, _ = tile_locs.shape
        centered_ptiles = self._get_images_in_centered_tiles(image_ptiles, tile_locs)
        assert centered_ptiles.shape[-1] == centered_ptiles.shape[-2] == self.slen
        x = rearrange(centered_ptiles, "ns np c h w -> (ns np) c h w")
        galaxy_params_flat, pq_divergence_flat = self.enc(x)
        galaxy_params = rearrange(
            galaxy_params_flat,
            "(ns np ms) d -> ns np ms d",
            ns=n_samples,
            np=n_ptiles,
            ms=max_sources,
        )
        if pq_divergence_flat.shape:
            pq_divergence = rearrange(
                pq_divergence_flat,
                "(ns np ms) -> ns np ms",
                ns=n_samples,
                np=n_ptiles,
                ms=max_sources,
            )
        else:
            pq_divergence = pq_divergence_flat
        return galaxy_params, pq_divergence

    def validation_epoch_end(self, outputs):
        """Pytorch lightning method run at end of validation epoch."""
        # put all outputs together into a single batch
        batch = {}
        for b in outputs:
            for k, v in b.items():
                curr_val = batch.get(k, torch.tensor([], device=v.device))
                batch[k] = torch.cat([curr_val, v])
        if self.n_bands == 1:
            self._make_plots(batch)

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
            "star_bools",
            "star_fluxes",
            "star_log_fluxes",
            "n_sources",
        ]
        for k in keys:
            batch[k] = batch[k][samples]

        # extract non-params entries so that 'get_full_params' to works.
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
        z, _ = self._encode(image_ptiles, locs)
        galaxy_params = rearrange(
            z,
            "ns (n nth ntw) ms d -> (ns n) nth ntw ms d",
            ns=1,
            nth=n_tiles_h,
            ntw=n_tiles_w,
        )

        tile_est = TileCatalog(
            self.tile_slen,
            {
                "n_sources": batch["n_sources"],
                "locs": batch["locs"],
                "galaxy_bools": batch["galaxy_bools"],
                "star_bools": batch["star_bools"],
                "star_fluxes": batch["star_fluxes"],
                "star_log_fluxes": batch["star_log_fluxes"],
                "galaxy_params": galaxy_params,
            },
        )
        est = tile_est.to_full_params()

        # draw all reconstruction images.
        # render_images automatically accounts for tiles with no galaxies.
        recon_images = self.image_decoder.render_images(tile_est)
        recon_images += background
        residuals = (images - recon_images) / torch.sqrt(recon_images)

        # draw worst `n_samples` examples as measured by absolute avg. residual error.
        worst_indices = residuals.abs().mean(dim=(1, 2, 3)).argsort(descending=True)[:n_samples]

        if self.crop_loss_at_border:
            bp = self.border_padding * 2
            residuals[:, :, :bp, :] = 0.0
            residuals[:, :, -bp:, :] = 0.0
            residuals[:, :, :, :bp] = 0.0
            residuals[:, :, :, -bp:] = 0.0

        figsize = (12, 4 * n_samples)
        fig, axes = plt.subplots(nrows=n_samples, ncols=3, figsize=figsize, squeeze=False)

        for i, idx in enumerate(worst_indices):

            true_ax = axes[i, 0]
            recon_ax = axes[i, 1]
            res_ax = axes[i, 2]

            # add titles to axes in the first row
            if i == 0:
                true_ax.set_title("Truth", size=18)
                recon_ax.set_title("Reconstruction", size=18)
                res_ax.set_title("Residual", size=18)

            # vmin, vmax should be shared between reconstruction and true images.
            if self.max_flux_valid_plots is None:
                vmax = np.ceil(max(images[idx].max().item(), recon_images[idx].max().item()))
            else:
                vmax = self.max_flux_valid_plots
            vmin = np.floor(min(images[idx].min().item(), recon_images[idx].min().item()))
            vrange = (vmin, vmax)

            # plot!
            labels = ("t. gal", None, "t. star", None)
            # Plotting only works on square images
            assert images.shape[-2] == images.shape[-1]
            slen = images.shape[-1] - 2 * self.border_padding
            bp = self.border_padding
            image = images[i, 0].cpu().numpy()
            plocs = est.plocs[i].cpu().numpy().reshape(-1, 2)
            probs = est["galaxy_bools"][i].cpu().numpy().reshape(-1)
            plot_image(fig, true_ax, image, vrange=vrange, colorbar=True)
            plot_locs(true_ax, bp, slen, plocs, probs, cmap="cool")
            plot_image(fig, recon_ax, image, vrange=vrange, colorbar=True)
            plot_locs(recon_ax, bp, slen, plocs, probs, cmap="cool")
            residuals_idx = residuals[idx, 0].cpu().numpy()
            res_vmax = np.ceil(residuals_idx.max())
            res_vmin = np.floor(residuals_idx.min())
            if self.crop_loss_at_border:
                bp = (recon_images.shape[-1] - slen) // 2
                eff_slen = slen - bp
                for b in (bp, bp * 2):
                    recon_ax.axvline(b, color="w")
                    recon_ax.axvline(b + eff_slen, color="w")
                    recon_ax.axhline(b, color="w")
                    recon_ax.axhline(b + eff_slen, color="w")
            plot_image(fig, res_ax, residuals_idx, vrange=(res_vmin, res_vmax))
            if self.crop_loss_at_border:
                for b in (bp, bp * 2):
                    res_ax.axvline(b, color="w")
                    res_ax.axvline(b + eff_slen, color="w")
                    res_ax.axhline(b, color="w")
                    res_ax.axhline(b + eff_slen, color="w")
            if i == 0:
                add_loc_legend(true_ax, labels)

        fig.tight_layout()
        if self.logger:
            self.logger.experiment.add_figure(
                f"Epoch:{self.current_epoch}/Worst_Validation_Images", fig
            )
        plt.close(fig)

    def _get_images_in_centered_tiles(self, image_ptiles: Tensor, tile_locs: Tensor) -> Tensor:
        n_bands = image_ptiles.shape[1] // 2
        img, bg = torch.split(image_ptiles, [n_bands, n_bands], dim=1)
        return self.center_ptiles(img - bg, tile_locs)

    def configure_optimizers(self):
        """Set up optimizers (pytorch-lightning method)."""
        return Adam(self.enc.parameters(), **self.optimizer_params)

    def forward(self, image_ptiles, tile_locs):
        raise NotImplementedError("Please use encode()")


class CenterPaddedTilesTransform(nn.Module):
    cached_grid: Tensor
    swap: Tensor

    def __init__(self, tile_slen: int, ptile_slen: int) -> None:
        super().__init__()
        self.tile_slen = tile_slen
        self.ptile_slen = ptile_slen
        self.bp = (self.ptile_slen - self.tile_slen) // 2
        self.register_buffer("cached_grid", get_mgrid(ptile_slen), persistent=False)
        self.register_buffer("swap", torch.tensor([1, 0]), persistent=False)

    def forward(self, image_ptiles: Tensor, tile_locs: Tensor) -> Tensor:
        n_ptiles, _, _, ptile_slen_img = image_ptiles.shape
        n_samples, n_ptiles_locs, max_sources, _ = tile_locs.shape
        assert max_sources == 1
        assert ptile_slen_img == self.ptile_slen
        assert n_ptiles_locs == n_ptiles

        # get new locs to do the shift
        ptile_locs = (tile_locs * self.tile_slen + self.bp) / self.ptile_slen
        offsets_hw = torch.tensor(1.0) - 2 * ptile_locs
        offsets_xy = offsets_hw.index_select(dim=-1, index=self.swap)
        grid_loc = self.cached_grid.view(1, self.ptile_slen, self.ptile_slen, 2) - offsets_xy.view(
            -1, 1, 1, 2
        )

        # Expand image_ptiles to match number of samples
        image_ptiles = image_ptiles.unsqueeze(0).expand(n_samples, -1, -1, -1, -1)
        image_ptiles = image_ptiles.reshape(
            n_samples * n_ptiles,
            -1,
            self.ptile_slen,
            self.ptile_slen,
        )
        shifted_tiles = F.grid_sample(image_ptiles, grid_loc, align_corners=True)

        # now that everything is center we can crop easily
        shifted_tiles = shifted_tiles[
            :,
            :,
            self.tile_slen : (self.ptile_slen - self.tile_slen),
            self.tile_slen : (self.ptile_slen - self.tile_slen),
        ]
        return rearrange(shifted_tiles, "(ns np) c h w -> ns np c h w", ns=n_samples, np=n_ptiles)
