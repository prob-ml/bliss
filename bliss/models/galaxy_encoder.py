import numpy as np
import pytorch_lightning as pl
import torch
from matplotlib import pyplot as plt
from torch.distributions import Normal
from torch.nn import functional as F

from bliss.models.decoder import ImageDecoder, get_mgrid
from bliss.models.encoder import get_full_params, get_images_in_tiles
from bliss.models.galaxy_net import CenteredGalaxyEncoder, OneCenteredGalaxyAE
from bliss.optimizer import get_optimizer
from bliss.plotting import plot_image, plot_image_and_locs


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
        hidden: int = 256,
        decoder_kwargs: dict = None,
        optimizer_params: dict = None,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.max_sources = 1  # by construction.

        # to produce images to train on.
        self.image_decoder = ImageDecoder(**decoder_kwargs)
        self.image_decoder.requires_grad_(False)

        # extract useful info from image_decoder
        self.latent_dim = self.image_decoder.n_galaxy_params
        self.n_bands = self.image_decoder.n_bands

        # put image dimensions together
        self.tile_slen = self.image_decoder.tile_slen
        self.border_padding = self.image_decoder.border_padding
        self.ptile_slen = self.tile_slen + 2 * self.border_padding
        self.slen = self.ptile_slen - 2 * self.tile_slen  # will always crop 2 * tile_slen

        # will be trained.
        # self.enc = CenteredGalaxyEncoder(
        #     slen=self.slen, latent_dim=self.latent_dim, n_bands=self.n_bands, hidden=hidden
        # )
        autoencoder_ckpt = decoder_kwargs["autoencoder_ckpt"]
        autoencoder = OneCenteredGalaxyAE.load_from_checkpoint(autoencoder_ckpt)
        # self.autoencoder.eval().requires_grad_(False)
        # autoencoder = OneCenteredGalaxyAE()
        self.enc = autoencoder.get_encoder(allow_pad=True)

        # grid for center cropped tiles
        self.register_buffer("cached_grid", get_mgrid(self.ptile_slen), persistent=False)
        self.register_buffer("swap", torch.tensor([1, 0]), persistent=False)

        # consistency
        assert self.image_decoder.max_sources == 1, "1 galaxy per tile is supported"
        assert self.slen >= 20, "Cropped slen is not reasonable for average sized galaxies."

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
        assert self.hparams["optimizer_params"] is not None, "Need to specify 'optimizer_params'."
        name = self.hparams["optimizer_params"]["name"]
        kwargs = self.hparams["optimizer_params"]["kwargs"]
        return get_optimizer(name, self.enc.parameters(), kwargs)

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

        # recon_losses = -Normal(recon_mean, recon_var.sqrt()).log_prob(images)
        # recon_losses = -Normal(recon_mean, recon_var.sqrt().clamp_min(1e-3)).log_prob(images)
        recon_losses = (recon_mean - images).pow(2)
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
    def make_plots(self, batch, n_samples=25):
        # validate worst reconstruction images.
        n_samples = min(len(batch["n_sources"]), n_samples)

        # extract non-params entries so that 'get_full_params' to works.
        exclude = {"images", "slen", "background"}
        images = batch["images"]
        slen = int(batch["slen"].unique().item())
        tile_params = {k: v for k, v in batch.items() if k not in exclude}

        # obtain map estimates
        tile_galaxy_params = self.forward_image(images, batch["locs"])
        tile_est = {
            k: (v if k != "galaxy_params" else tile_galaxy_params) for k, v in tile_params.items()
        }
        est = get_full_params(tile_est, slen)

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
        res_vmax = torch.ceil(residuals[worst_indices].max().cpu()).item()
        res_vmin = torch.floor(residuals[worst_indices].min().cpu()).item()

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
            plot_image(fig, res_ax, residuals[idx, 0].cpu().numpy(), vrange=(res_vmin, res_vmax))

        fig.tight_layout()
        if self.logger:
            self.logger.experiment.add_figure(f"Epoch:{self.current_epoch}/Validation Images", fig)
        plt.close(fig)
