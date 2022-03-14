from pathlib import Path
from typing import Optional, Union

import numpy as np
import pytorch_lightning as pl
import torch
from einops import rearrange
from matplotlib import pyplot as plt
from torch import Tensor
from torch.distributions import Normal
from torch.nn import functional as F
from torch.optim import Adam

from bliss.models.decoder import ImageDecoder, get_mgrid
from bliss.models.galaxy_net import OneCenteredGalaxyAE
from bliss.models.location_encoder import get_full_params_from_tiles, get_images_in_tiles
from bliss.models.vae.galaxy_flow import CenteredGalaxyLatentFlow
from bliss.models.vae.galaxy_vae import OneCenteredGalaxyVAE
from bliss.reporting import plot_image, plot_image_and_locs


class GalaxyEncoder(pl.LightningModule):
    def __init__(
        self,
        decoder: ImageDecoder,
        autoencoder: Union[OneCenteredGalaxyAE, OneCenteredGalaxyVAE],
        hidden: int,
        vae_flow: Optional[CenteredGalaxyLatentFlow] = None,
        vae_flow_ckpt: Optional[str] = None,
        optimizer_params: dict = None,
        crop_loss_at_border=False,
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
            vae_flow.load_state_dict(torch.load(vae_flow_ckpt, map_location=vae_flow.device))
            vae_flow.eval()
            vae_flow.requires_grad_(False)
            self.enc.p_z = vae_flow
        self.latent_dim = autoencoder.latent_dim

        # grid for center cropped tiles
        self.register_buffer("cached_grid", get_mgrid(self.ptile_slen), persistent=False)
        self.register_buffer("swap", torch.tensor([1, 0]), persistent=False)

        # consistency
        assert self.slen >= 20, "Cropped slen is not reasonable for average sized galaxies."

        if checkpoint_path is not None:
            self.load_state_dict(
                torch.load(Path(checkpoint_path), map_location=torch.device("cpu"))
            )

    def encode(self, images: Tensor, background: Tensor, tile_locs: Tensor) -> Tensor:
        """Runs galaxy encoder on input image ptiles (with bg substracted)."""
        batch_size, nth, ntw, max_sources, _ = tile_locs.shape
        centered_ptiles = self._get_images_in_centered_tiles(images, background, tile_locs)
        assert centered_ptiles.shape[-1] == centered_ptiles.shape[-2] == self.slen
        z_flat, pq_z_flat = self.enc(centered_ptiles)
        z = rearrange(
            z_flat,
            "(b nth ntw s) d -> b nth ntw s d",
            b=batch_size,
            nth=ntw,
            ntw=ntw,
            s=max_sources,
        )
        if pq_z_flat.shape:
            pq_z = rearrange(
                pq_z_flat,
                "(b nth ntw s) -> b nth ntw s",
                b=batch_size,
                nth=nth,
                ntw=ntw,
                s=max_sources,
            )
        else:
            pq_z = pq_z_flat
        return z, pq_z

    def sample(self, images, background, tile_locs):
        z, _ = self.encode(images, background, tile_locs)
        return z

    def max_a_post(self, images: Tensor, background: Tensor, tile_locs: Tensor) -> Tensor:
        # assert image_ptiles.shape[-1] == image_ptiles.shape[-2] == self.ptile_slen
        # batch_size, n_tiles_h, n_tiles_w, _, _, _ = image_ptiles.shape

        # centered_ptiles = self._flatten_and_center_ptiles(image_ptiles, tile_locs)
        # assert centered_ptiles.shape[-1] == centered_ptiles.shape[-2] == self.slen
        # # We can assume there is one galaxy per_tile and encode each tile independently.
        batch_size, nth, ntw, max_sources, _ = tile_locs.shape
        centered_ptiles = self._get_images_in_centered_tiles(images, background, tile_locs)
        z_flat = self.enc.max_a_post(centered_ptiles)
        return rearrange(
            z_flat,
            "(b nth ntw s) d -> b nth ntw s d",
            b=batch_size,
            nth=nth,
            ntw=ntw,
            s=max_sources,
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
        images = batch["images"]
        background = batch["background"]
        tile_locs = batch["locs"]
        z, pq_z = self.encode(images, background, tile_locs)
        # draw fully reconstructed image.
        # NOTE: Assume recon_mean = recon_var per poisson approximation.
        recon_mean = self.image_decoder.render_images(
            batch["n_sources"],
            batch["locs"],
            batch["galaxy_bools"],
            z,
            batch["fluxes"],
        )
        recon_mean += background

        assert not torch.any(torch.isnan(recon_mean))
        assert not torch.any(torch.isinf(recon_mean))
        recon_losses = -Normal(recon_mean, recon_mean.sqrt()).log_prob(images)
        if self.crop_loss_at_border:
            bp = self.border_padding * 2
            recon_losses = recon_losses[:, :, bp:(-bp), bp:(-bp)]
        assert not torch.any(torch.isnan(recon_losses))
        assert not torch.any(torch.isinf(recon_losses))

        # For divergence loss, we only evaluate tiles with a galaxy in them
        divergence_loss = (pq_z.unsqueeze(-1) * batch["galaxy_bools"]).sum()
        return recon_losses.sum() - divergence_loss

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
            "fluxes",
            "log_fluxes",
            "n_sources",
        ]
        for k in keys:
            batch[k] = batch[k][samples]

        # extract non-params entries so that 'get_full_params' to works.
        images = batch["images"]
        background = batch["background"]
        tile_locs = batch["locs"]

        # obtain map estimates
        z, _ = self.encode(images, background, tile_locs)

        tile_est = {
            "n_sources": batch["n_sources"],
            "locs": batch["locs"],
            "galaxy_bools": batch["galaxy_bools"],
            "star_bools": batch["star_bools"],
            "fluxes": batch["fluxes"],
            "log_fluxes": batch["log_fluxes"],
            "galaxy_params": z,
        }
        est = get_full_params_from_tiles(tile_est, self.tile_slen)

        # draw all reconstruction images.
        # render_images automatically accounts for tiles with no galaxies.
        recon_images = self.image_decoder.render_images(
            tile_est["n_sources"],
            tile_est["locs"],
            tile_est["galaxy_bools"],
            tile_est["galaxy_params"],
            tile_est["fluxes"],
        )
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
            labels = None if i > 0 else ("t. gal", None, "t. star", None)
            # Plotting only works on square images
            assert images.shape[-2] == images.shape[-1]
            slen = images.shape[-1] - 2 * self.border_padding
            plot_image_and_locs(idx, fig, true_ax, images, slen, est, labels=labels, vrange=vrange)
            plot_image_and_locs(
                idx, fig, recon_ax, recon_images, slen, est, labels=labels, vrange=vrange
            )
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

        fig.tight_layout()
        if self.logger:
            self.logger.experiment.add_figure(f"Epoch:{self.current_epoch}/Validation Images", fig)
        plt.close(fig)

    def _get_images_in_centered_tiles(
        self, images: Tensor, background: Tensor, tile_locs: Tensor
    ) -> Tensor:
        image_ptiles = get_images_in_tiles(
            images - background, tile_slen=self.tile_slen, ptile_slen=self.ptile_slen
        )
        return self._flatten_and_center_ptiles(image_ptiles, tile_locs)

    def _flatten_and_center_ptiles(self, image_ptiles, tile_locs):
        image_ptiles_flat = rearrange(image_ptiles, "b nth ntw c h w -> (b nth ntw) c h w")
        tile_locs_flat = rearrange(tile_locs, "b nth ntw s xy -> (b nth ntw) s xy")
        return center_ptiles(
            image_ptiles_flat,
            tile_locs_flat,
            self.tile_slen,
            self.ptile_slen,
            self.border_padding,
            self.swap,
            self.cached_grid,
        )

    def configure_optimizers(self):
        """Set up optimizers (pytorch-lightning method)."""
        return Adam(self.enc.parameters(), **self.optimizer_params)

    def forward(self, image_ptiles, tile_locs):
        raise NotImplementedError("Please use encode()")


def center_ptiles(
    image_ptiles, tile_locs, tile_slen, ptile_slen, border_padding, swap, cached_grid
):
    # assume there is at most one source per tile
    # return a centered version of sources in tiles using their true locations in tiles.
    # also we crop them to avoid sharp borders with no bacgkround/noise.

    # round up necessary variables and paramters
    assert len(image_ptiles.shape) == 4
    assert len(tile_locs.shape) == 3
    assert tile_locs.shape[1] == 1
    assert image_ptiles.shape[-1] == ptile_slen
    n_ptiles = image_ptiles.shape[0]
    assert tile_locs.shape[0] == n_ptiles

    # get new locs to do the shift
    ptile_locs = tile_locs * tile_slen + border_padding
    ptile_locs /= ptile_slen
    locs0 = torch.tensor([ptile_slen - 1, ptile_slen - 1]) / 2
    locs0 /= ptile_slen - 1
    locs0 = locs0.view(1, 1, 2).to(image_ptiles.device)
    locs = 2 * locs0 - ptile_locs

    # center tiles on the corresponding source given by locs.
    locs = (locs - 0.5) * 2
    locs = locs.index_select(2, swap)  # transpose (x,y) coords
    grid_loc = cached_grid.view(1, ptile_slen, ptile_slen, 2) - locs.view(-1, 1, 1, 2)
    shifted_tiles = F.grid_sample(image_ptiles, grid_loc, align_corners=True)

    # now that everything is center we can crop easily
    return shifted_tiles[
        :, :, tile_slen : (ptile_slen - tile_slen), tile_slen : (ptile_slen - tile_slen)
    ]
