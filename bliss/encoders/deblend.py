from copy import deepcopy

import pytorch_lightning as pl
import torch
from einops import pack, rearrange, unpack
from torch import Tensor
from torch.distributions import Normal
from torch.optim import Adam

from bliss.catalog import TileCatalog
from bliss.datasets.galsim_blends import parse_dataset
from bliss.encoders.autoencoder import CenteredGalaxyEncoder, OneCenteredGalaxyAE
from bliss.grid import shift_sources_in_ptiles, validate_border_padding
from bliss.render_tiles import (
    get_images_in_tiles,
    reconstruct_image_from_ptiles,
    render_galaxy_ptiles,
)


class GalaxyEncoder(pl.LightningModule):
    def __init__(
        self,
        ae_state_dict_path: str,
        n_bands: int = 1,
        tile_slen: int = 4,
        ptile_slen: int = 52,
        decoder_slen: int = 53,
        latent_dim: int = 8,
        hidden: int = 256,
        lr: float = 1e-4,
    ):
        super().__init__()

        # dimensions
        self.n_bands = n_bands
        self.tile_slen = tile_slen
        self.ptile_slen = ptile_slen
        self.bp = validate_border_padding(tile_slen, ptile_slen)
        self.final_slen = self.ptile_slen - 2 * self.tile_slen  # will always crop 2 * tile_slen

        self.lr = lr

        # encoder (to be trained)
        self._latent_dim = latent_dim
        self._hidden = hidden
        self._enc = CenteredGalaxyEncoder(self.final_slen, latent_dim, n_bands, hidden)

        # decoder
        ae = OneCenteredGalaxyAE(decoder_slen, latent_dim, hidden, n_bands)
        ae.load_state_dict(torch.load(ae_state_dict_path))
        self._dec = deepcopy(ae.dec)
        self._dec.requires_grad_(False)
        self._dec.eval()
        del ae

    def forward(self, image_ptiles_flat: Tensor, tile_locs_flat: Tensor) -> Tensor:
        """Runs galaxy encoder on input image ptiles (with bg substracted)."""
        centered_ptiles = self._get_centered_padded_tiles(image_ptiles_flat, tile_locs_flat)
        return self._enc(centered_ptiles)

    def get_loss(
        self, images: Tensor, paddings: Tensor, background: Tensor, tile_catalog: TileCatalog
    ):
        _, nth, ntw, _ = tile_catalog.locs.shape

        images_with_background, _ = pack([images, background], "b * h w")
        image_ptiles = get_images_in_tiles(
            images_with_background,
            self.tile_slen,
            self.ptile_slen,
        )
        image_ptiles_flat = rearrange(image_ptiles, "n nth ntw c h w -> (n nth ntw) c h w")
        tile_locs_flat = rearrange(tile_catalog.locs, "n nth ntw xy -> (n nth ntw) xy")
        galaxy_params_flat: Tensor = self(image_ptiles_flat, tile_locs_flat)
        assert galaxy_params_flat.ndim == 2

        # draw fully reconstructed image.
        # NOTE: Assume recon_mean = recon_var per poisson approximation.
        galaxy_params_pred = rearrange(
            galaxy_params_flat, "(b nth ntw) d -> b nth ntw d", nth=nth, ntw=ntw
        )
        recon_ptiles = render_galaxy_ptiles(
            self._dec,
            tile_catalog.locs,
            galaxy_params_pred,
            tile_catalog["galaxy_bools"],
            self.ptile_slen,
            self.tile_slen,
            self.n_bands,
        )
        assert recon_ptiles.shape[-1] == recon_ptiles.shape[-2] == self.ptile_slen
        recon_mean = reconstruct_image_from_ptiles(recon_ptiles, self.tile_slen)
        recon_mean += background + paddings  # target only galaxies within tiles.
        assert recon_mean.ndim == 4 and recon_mean.shape[-1] == images.shape[-1]
        assert not torch.any(torch.logical_or(torch.isnan(recon_mean), torch.isinf(recon_mean)))

        recon_losses: Tensor = -Normal(recon_mean, recon_mean.sqrt()).log_prob(images)
        assert not torch.any(torch.logical_or(torch.isnan(recon_losses), torch.isinf(recon_losses)))

        return recon_losses.mean()

    def variational_mode(self, images: Tensor, background: Tensor, tile_catalog: TileCatalog):
        _, nth, ntw, _ = tile_catalog.locs.shape

        images_with_background, _ = pack([images, background], "b * h w")
        image_ptiles = get_images_in_tiles(
            images_with_background,
            self.tile_slen,
            self.ptile_slen,
        )
        image_ptiles_flat = rearrange(image_ptiles, "n nth ntw c h w -> (n nth ntw) c h w")
        tile_locs_flat = rearrange(tile_catalog.locs, "n nth ntw xy -> (n nth ntw) xy")
        galaxy_params_flat: Tensor = self(image_ptiles_flat, tile_locs_flat)

        return rearrange(galaxy_params_flat, "(b nth ntw) d -> b nth ntw d", nth=nth, ntw=ntw)

    def training_step(self, batch, batch_idx):
        """Pytorch lightning training step."""
        images, background, tile_catalog, paddings = parse_dataset(batch, self.tile_slen)
        loss = self.get_loss(images, paddings, background, tile_catalog)
        self.log("train/loss", loss, batch_size=len(images))
        return loss

    def validation_step(self, batch, batch_idx):
        """Pytorch lightning validation step."""
        images, background, tile_catalog, paddings = parse_dataset(batch, self.tile_slen)
        loss = self.get_loss(images, paddings, background, tile_catalog)
        self.log("val/loss", loss, batch_size=len(images))
        return loss

    def configure_optimizers(self):
        """Set up optimizers."""
        return Adam(self._enc.parameters(), self.lr)

    def _get_centered_padded_tiles(self, image_ptiles: Tensor, tile_locs_flat: Tensor) -> Tensor:
        """Remove background, center padded tiles at given locations, and crop."""
        img, bg = unpack(image_ptiles, [(1,), (1,)], "npt * h w")
        shifted_ptiles = shift_sources_in_ptiles(
            img - bg, tile_locs_flat, self.tile_slen, self.ptile_slen, center=True
        )
        assert shifted_ptiles.shape[-1] == shifted_ptiles.shape[-2] == self.ptile_slen
        cropped_ptiles = shifted_ptiles[
            ...,
            self.tile_slen : (self.ptile_slen - self.tile_slen),
            self.tile_slen : (self.ptile_slen - self.tile_slen),
        ]
        assert cropped_ptiles.shape[-1] == cropped_ptiles.shape[-2] == self.final_slen
        return cropped_ptiles
