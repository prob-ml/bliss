from typing import Dict, List

import pytorch_lightning as pl
import torch
from einops import rearrange
from matplotlib import pyplot as plt
from torch import Tensor
from torch.distributions import Normal
from torch.nn import functional as F
from torch.optim import Adam

from bliss.catalog import TileCatalog, get_images_in_tiles
from bliss.models.decoder import get_mgrid
from bliss.models.galaxy_encoder import center_ptiles
from bliss.models.location_encoder import EncoderCNN, make_enc_final

class GalaxyFluxEncoder(pl.LightningModule):
    def __init__(
        self, 
        input_transform,
        n_bands,
        tile_slen,
        ptile_slen,
        channel: int,
        hidden: int,
        spatial_dropout: float,
        dropout: float,
        optimizer_params: dict = None,
    ) -> None:
        super().__init__()
        self.input_transform = input_transform
        self.n_bands = n_bands
        self.tile_slen = tile_slen
        self.ptile_slen = ptile_slen
        border_padding = (ptile_slen - tile_slen) / 2
        assert tile_slen <= ptile_slen
        assert border_padding % 1 == 0, "amount of border padding should be an integer"
        self.border_padding = int(border_padding)
        self.slen = self.ptile_slen - 2 * self.tile_slen  # will always crop 2 * tile_slen

        self.optimizer_params = optimizer_params

        dim_enc_conv_out = ((self.slen + 1) // 2 + 1) // 2
        n_bands_in = self.input_transform.output_channels(n_bands)
        self.enc_conv = EncoderCNN(n_bands_in, channel, spatial_dropout)
        self.enc_final = make_enc_final(channel * 4 * dim_enc_conv_out ** 2, hidden, 2, dropout)

        # grid for center cropped tiles
        self.register_buffer("cached_grid", get_mgrid(self.ptile_slen), persistent=False)
        self.register_buffer("swap", torch.tensor([1, 0]), persistent=False)

    def encode(self, images: Tensor, background: Tensor, locs: Tensor) -> Tensor:
        centered_tiles = self._get_images_in_centered_tiles(images, background, locs)
        assert centered_tiles.shape[-1] == centered_tiles.shape[-2] == self.slen
        h = self.enc_conv(centered_tiles)
        h2 = self.enc_final(h)
        batch_size, n_tiles_h, n_tiles_w, max_sources, _ = locs.shape
        return rearrange(
            h2, 
            "(b nth ntw s) d -> b nth ntw s d",
            b=batch_size,
            nth=n_tiles_h,
            ntw=n_tiles_w,
            s=max_sources,
            d=self.n_bands * 2
        )

    def max_a_post(self, images, background, locs):
        return self.encode(images, background, locs)[:, :, :, :, 0:1]

    def get_loss(self, batch: Dict[str, Tensor]):
        images = batch["images"]
        background = batch["background"]
        tile_catalog = TileCatalog(self.tile_slen, {k:v for k, v in batch.items() if k not in {"images", "background"}})
        galaxy_log_flux_params = self.encode(images, background, tile_catalog.locs)
        
        true_log_fluxes = tile_catalog["galaxy_params"][:, :, :, :, -self.n_bands:]
        galaxy_log_flux_mean, galaxy_log_flux_logvar = torch.split(galaxy_log_flux_params, (self.n_bands, self.n_bands), -1)
        # galaxy_log_flux_mean.clamp(min=)
        # galaxy_log_flux_mean =  6.0 + F.softplus(galaxy_log_flux_mean)
        # galaxy_log_flux_mean = galaxy_log_flux_mean.clamp_max(14.0)
        galaxy_log_flux_dist = Normal(galaxy_log_flux_mean, torch.exp(galaxy_log_flux_logvar * 0.5) + 1e-3)
        log_probs = galaxy_log_flux_dist.log_prob(true_log_fluxes)
        # log_probs = -torch.pow(galaxy_log_flux_mean - true_log_fluxes, 2)
        # return (-log_probs * tile_catalog["galaxy_bools"]).sum(), galaxy_log_flux_params, tile_catalog
        return (-log_probs[tile_catalog["galaxy_bools"] > 0.5]).sum(), galaxy_log_flux_params, tile_catalog
    
    def training_step(self, batch, batch_idx):
        loss, _, _ = self.get_loss(batch)
        self.log("train/loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, galaxy_log_flux_params, tile_catalog = self.get_loss(batch)
        self.log("val/loss", loss)
        output = ({"images": batch["images"], "background": batch["background"], "loss": loss, "galaxy_log_flux_params": galaxy_log_flux_params, "tile_catalog": tile_catalog})
        return output

    def validation_epoch_end(self, outputs: List[Dict[str, Tensor]], n_samples = 9):
        assert n_samples ** (0.5) % 1 == 0
        if n_samples > len(outputs):  # do nothing if low on samples.
            return
        nrows = int(n_samples ** 0.5)  # for figure
        fig, axes = plt.subplots(nrows=nrows, ncols=nrows, figsize=(12, 12))
        axes = axes.flatten()

        for i in range(n_samples):
            batch = outputs[i]
            tile_catalog: TileCatalog = batch["tile_catalog"]
            galaxy_log_flux_mean, _ = torch.split(batch["galaxy_log_flux_params"], (self.n_bands, self.n_bands), -1)
            # galaxy_log_flux_mean = 6.0 + F.softplus(galaxy_log_flux_mean)
            tile_catalog["galaxy_log_fluxes"] = galaxy_log_flux_mean * tile_catalog["galaxy_bools"]
            full_catalog = tile_catalog.to_full_params()
            ax = axes[i]
            ax.imshow(batch["images"][0,0].cpu().numpy(), cmap="gist_gray")
            plocs = full_catalog.plocs[0, :, :] + self.border_padding - 0.5
            ax.scatter(plocs[:, 1].cpu(), plocs[:, 0].cpu())

            galaxy_bools = full_catalog["galaxy_bools"][0, :, 0] > 0.5
            plocs_galaxies = plocs[galaxy_bools]
            true_galaxy_fluxes = full_catalog["galaxy_params"][0, galaxy_bools][:, -self.n_bands:]
            est_galaxy_fluxes = full_catalog["galaxy_log_fluxes"][0, galaxy_bools]
            for (i, ploc) in enumerate(plocs_galaxies):
                xi = ploc[1]
                yi = ploc[0]
                ax.annotate(f"True:{true_galaxy_fluxes[i, 0]:.2e}.\nEst:{est_galaxy_fluxes[i, 0]:.2e}", (xi, yi), color = "c")
        fig.tight_layout()

        title = f"Epoch:{self.current_epoch}/Validation Images"
        if self.logger is not None:
            self.logger.experiment.add_figure(title, fig)

    def _get_images_in_centered_tiles(
        self, images: Tensor, background: Tensor, tile_locs: Tensor
    ) -> Tensor:
        """Divide a batch of full images into padded tiles similar to nn.conv2d."""
        log_image_ptiles = get_images_in_tiles(
            self.input_transform(images, background),
            self.tile_slen,
            self.ptile_slen,
        )
        assert log_image_ptiles.shape[-1] == log_image_ptiles.shape[-2] == self.ptile_slen
        # in each padded tile we need to center the corresponding galaxy/star
        return self._flatten_and_center_ptiles(log_image_ptiles, tile_locs)

    def _flatten_and_center_ptiles(self, log_image_ptiles, tile_locs):
        """Return padded tiles centered on each corresponding source."""
        log_image_ptiles_flat = rearrange(log_image_ptiles, "b nth ntw c h w -> (b nth ntw) c h w")
        tile_locs_flat = rearrange(tile_locs, "b nth ntw s xy -> (b nth ntw) s xy")
        return center_ptiles(
            log_image_ptiles_flat,
            tile_locs_flat,
            self.tile_slen,
            self.ptile_slen,
            self.border_padding,
            self.swap,
            self.cached_grid,
        )

    def configure_optimizers(self):
        """Pytorch lightning method."""
        return Adam(self.parameters(), **self.optimizer_params)


    
    def forward(self, x):
        raise NotImplementedError()