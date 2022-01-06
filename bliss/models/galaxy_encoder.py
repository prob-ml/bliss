from typing import Optional

import numpy as np
import pytorch_lightning as pl
import torch
from matplotlib import pyplot as plt
from torch.distributions import Normal
from torch.nn import functional as F

from bliss.models.prior import ImagePrior
from bliss.models.decoder import ImageDecoder, get_mgrid
from bliss.models.location_encoder import get_images_in_tiles, get_full_params_from_tiles
from bliss.models.galaxy_net import OneCenteredGalaxyAE
from bliss.optimizer import load_optimizer
from bliss.reporting import plot_image, plot_image_and_locs


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


class GalaxyEncoder(pl.LightningModule):
    def __init__(
        self,
        prior,
        decoder,
        hidden: int = 256,
        optimizer_params: dict = None,
        crop_loss_at_border=False,
        checkpoint_path: Optional[str] = None,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.max_sources = 1  # by construction.
        self.crop_loss_at_border = crop_loss_at_border

        # to produce images to train on.
        self.image_prior = ImagePrior(**prior)
        self.image_decoder = ImageDecoder(**decoder)
        self.image_decoder.requires_grad_(False)

        # extract useful info from image_decoder
        self.n_bands = self.image_decoder.n_bands
        self.decoder_slen = self.image_decoder.slen

        # put image dimensions together
        self.tile_slen = self.image_decoder.tile_slen
        self.border_padding = self.image_decoder.border_padding
        self.ptile_slen = self.tile_slen + 2 * self.border_padding
        self.slen = self.ptile_slen - 2 * self.tile_slen  # will always crop 2 * tile_slen

        # will be trained.
        autoencoder_ckpt = decoder["autoencoder_ckpt"]
        autoencoder = OneCenteredGalaxyAE.load_from_checkpoint(autoencoder_ckpt)
        self.enc = autoencoder.get_encoder(allow_pad=True)
        self.latent_dim = autoencoder.latent_dim

        # grid for center cropped tiles
        self.register_buffer("cached_grid", get_mgrid(self.ptile_slen), persistent=False)
        self.register_buffer("swap", torch.tensor([1, 0]), persistent=False)

        # consistency
        assert self.slen >= 20, "Cropped slen is not reasonable for average sized galaxies."

        if checkpoint_path is not None:
            ge = GalaxyEncoder.load_from_checkpoint(checkpoint_path)
            self.load_state_dict(ge.state_dict())

    def center_ptiles(self, image_ptiles, tile_locs):
        return center_ptiles(
            image_ptiles,
            tile_locs,
            self.tile_slen,
            self.ptile_slen,
            self.border_padding,
            self.swap,
            self.cached_grid,
        )

    def configure_optimizers(self):
        """Set up optimizers (pytorch-lightning method)."""
        return load_optimizer(self.enc.parameters(), self.hparams)

    def forward_image(self, images, tile_locs):
        batch_size = images.shape[0]
        ptiles = get_images_in_tiles(images, self.tile_slen, self.ptile_slen)
        galaxy_params = self(ptiles, tile_locs)
        return galaxy_params.view(batch_size, -1, 1, self.latent_dim)

    def forward(self, image_ptiles, tile_locs):
        """Runs galaxy encoder on input image ptiles."""
        assert image_ptiles.shape[-1] == image_ptiles.shape[-2] == self.ptile_slen
        n_ptiles = image_ptiles.shape[0]

        # in each padded tile we need to center the corresponding galaxy
        tile_locs = tile_locs.reshape(n_ptiles, self.max_sources, 2)
        centered_ptiles = self.center_ptiles(image_ptiles, tile_locs)
        assert centered_ptiles.shape[-1] == centered_ptiles.shape[-2] == self.slen

        # remove background before encoding
        ptile_background = self.image_decoder.get_background(self.slen)
        centered_ptiles -= ptile_background.unsqueeze(0)

        # We can assume there is one galaxy per_tile and encode each tile independently.
        z = self.enc(centered_ptiles)
        assert z.shape[0] == n_ptiles
        return z

    def get_loss(self, batch):
        images = batch["images"]
        tile_galaxy_params = self.forward_image(images, batch["locs"])

        # draw fully reconstructed image.
        # NOTE: Assume recon_mean = recon_var per poisson approximation.
        recon_mean, recon_var = self.image_decoder.render_images(
            batch["n_sources"],
            batch["locs"],
            batch["galaxy_bool"],
            tile_galaxy_params,
            batch["fluxes"],
            add_noise=False,
        )

        recon_losses = -Normal(recon_mean, recon_var.sqrt()).log_prob(images)
        if self.crop_loss_at_border:
            slen = batch["slen"].unique().item()
            bp = (recon_losses.shape[-1] - slen) // 2
            bp = bp * 2
            recon_losses = recon_losses[:, :, bp:(-bp), bp:(-bp)]
        return recon_losses.sum()

    def training_step(self, batch, batch_idx):
        """Pytorch lightning training step."""
        loss = self.get_loss(batch)
        self.log("train/loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        """Pytorch lightning validation step."""
        loss = self.get_loss(batch)
        self.log("val/loss", loss)
        return batch

    def validation_epoch_end(self, outputs):
        """Pytorch lightning method run at end of validation epoch."""
        # put all outputs together into a single batch
        batch = {}
        for b in outputs:
            for k, v in b.items():
                curr_val = batch.get(k, torch.tensor([], device=v.device))
                batch[k] = torch.cat([curr_val, v])
        if self.n_bands == 1:
            self.make_plots(batch)

    # pylint: disable=too-many-statements
    def make_plots(self, batch, n_samples=5):
        # validate worst reconstruction images.
        for k in batch.keys():
            print(f"{k}: {batch[k].shape}")
        print(f"n_samples: {n_samples}")
        n_samples = min(len(batch["n_sources"]), n_samples)
        print(f"n_samples: {n_samples}")
        samples = np.random.choice(len(batch["n_sources"]), n_samples, replace=False)
        keys = [
            "images",
            "locs",
            "galaxy_bool",
            "star_bool",
            "fluxes",
            "log_fluxes",
            "n_sources",
        ]
        for k in keys:
            batch[k] = batch[k][samples]

        # extract non-params entries so that 'get_full_params' to works.
        images = batch["images"]
        slen = int(batch["slen"].unique().item())
        # obtain map estimates
        tile_galaxy_params = self.forward_image(images, batch["locs"])

        tile_est = {
            "n_sources": batch["n_sources"],
            "locs": batch["locs"],
            "galaxy_bool": batch["galaxy_bool"],
            "star_bool": batch["star_bool"],
            "fluxes": batch["fluxes"],
            "log_fluxes": batch["log_fluxes"],
            "galaxy_params": tile_galaxy_params,
        }
        est = get_full_params_from_tiles(tile_est, self.tile_slen)

        # draw all reconstruction images.
        # render_images automatically accounts for tiles with no galaxies.
        recon_images, _ = self.image_decoder.render_images(
            tile_est["n_sources"],
            tile_est["locs"],
            tile_est["galaxy_bool"],
            tile_est["galaxy_params"],
            tile_est["fluxes"],
            add_noise=False,
        )
        residuals = (images - recon_images) / torch.sqrt(recon_images)

        # draw worst `n_samples` examples as measured by absolute avg. residual error.
        worst_indices = residuals.abs().mean(dim=(1, 2, 3)).argsort(descending=True)[:n_samples]

        # use same vmin, vmax throughout for residuals
        if self.crop_loss_at_border:
            bp = (recon_images.shape[-1] - slen) // 2
            bp = bp * 2
            residuals[:, :, :bp, :] = 0.0
            residuals[:, :, -bp:, :] = 0.0
            residuals[:, :, :, :bp] = 0.0
            residuals[:, :, :, -bp:] = 0.0

        figsize = (12, 4 * n_samples)
        fig, axes = plt.subplots(nrows=n_samples, ncols=3, figsize=figsize)

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
            vmax = np.ceil(max(images[idx].max().item(), recon_images[idx].max().item()))
            vmin = np.floor(min(images[idx].min().item(), recon_images[idx].min().item()))
            vrange = (vmin, vmax)

            # plot!
            labels = None if i > 0 else ("t. gal", None, "t. star", None)
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
