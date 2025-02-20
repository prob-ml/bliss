import pytorch_lightning as pl
import torch
from einops import rearrange, reduce
from torch import Tensor
from torch.nn import BCELoss
from torch.optim import Adam

from bliss.catalog import TileCatalog
from bliss.datasets.generate_blends import parse_dataset
from bliss.encoders.layers import EncoderCNN, make_enc_final
from bliss.grid import shift_sources_bilinear, validate_border_padding
from bliss.render_tiles import get_images_in_tiles


class BinaryEncoder(pl.LightningModule):
    def __init__(
        self,
        n_bands: int = 1,
        tile_slen: int = 5,
        ptile_slen: int = 53,
        channel: int = 8,
        hidden: int = 128,
        spatial_dropout: float = 0,
        dropout: float = 0,
    ):
        """Encoder which conditioned on other source params returns probability of galaxy vs. star.

        This class implements the binary encoder, which takes in a synthetic image
        along with true locations and source parameters and returns whether each source in that
        image is a star or a galaxy.

        Arguments:
            n_bands: number of bands
            tile_slen: dimension (in pixels) of each tile.
            ptile_slen: dimension (in pixels) of the individual image padded tiles.
            channel: TODO (document this)
            hidden: TODO (document this)
            spatial_dropout: TODO (document this)
            dropout: TODO (document this)
        """
        super().__init__()
        self.save_hyperparameters()

        # extract useful info from image_decoder
        self.n_bands = n_bands

        # put image dimensions together
        self.tile_slen = tile_slen
        self.ptile_slen = ptile_slen
        self.bp = validate_border_padding(tile_slen, ptile_slen)
        self.final_slen = self.ptile_slen - 2 * self.tile_slen  # will always crop 2 * tile_slen

        dim_enc_conv_out = ((self.final_slen + 1) // 2 + 1) // 2
        self._enc_conv = EncoderCNN(n_bands, channel, spatial_dropout)
        self._enc_final = make_enc_final(channel * 4 * dim_enc_conv_out**2, hidden, 1, dropout)

    def forward(self, images: Tensor, locs: Tensor) -> Tensor:
        """Runs the binary encoder on centered_ptiles."""
        flat_locs = rearrange(locs, "n nth ntw xy -> (n nth ntw) xy", xy=2)

        image_ptiles = get_images_in_tiles(images, self.tile_slen, self.ptile_slen)
        image_ptiles_flat = rearrange(image_ptiles, "n nth ntw c h w -> (n nth ntw) c h w")

        return self.encode_tiled(image_ptiles_flat, flat_locs)

    def encode_tiled(self, image_ptiles_flat: Tensor, flat_locs: Tensor):
        npt, _ = flat_locs.shape
        centered_tiles = self._center_ptiles(image_ptiles_flat, flat_locs)
        x = rearrange(centered_tiles, "npt c h w -> npt c h w")
        h = self._enc_conv(x)
        h2 = self._enc_final(h)
        galaxy_probs = torch.sigmoid(h2).clamp(1e-4, 1 - 1e-4)
        return rearrange(galaxy_probs, "npt 1 -> npt", npt=npt)

    def get_loss(self, images: Tensor, tile_catalog: TileCatalog):
        """Return loss, accuracy, binary probabilities, and MAP classifications for given batch."""
        b, nth, ntw, _ = tile_catalog.locs.shape

        n_sources = tile_catalog.n_sources
        locs = tile_catalog.locs
        galaxy_bools = tile_catalog["galaxy_bools"]

        n_sources_flat = rearrange(n_sources, "b nth ntw -> (b nth ntw)")
        galaxy_bools_flat = rearrange(galaxy_bools, "b nth ntw 1 -> (b nth ntw 1)")

        galaxy_probs_flat: Tensor = self(images, locs)

        # accuracy
        with torch.no_grad():
            hits = galaxy_probs_flat.ge(0.5).eq(galaxy_bools_flat.bool())
            hits_with_one_source = hits.logical_and(n_sources_flat.eq(1))
            acc = hits_with_one_source.sum() / n_sources_flat.sum()

        # we need to calculate cross entropy loss, only for "on" sources
        raw_loss = BCELoss(reduction="none")(galaxy_probs_flat, galaxy_bools_flat.float())
        loss_vec = raw_loss * n_sources_flat.float()

        # as per paper, we sum over tiles and take mean over batches
        loss_per_tile = rearrange(loss_vec, "(b nth ntw) -> b nth ntw", b=b, nth=nth, ntw=ntw)
        loss_per_batch = reduce(loss_per_tile, "b nth ntw -> b", "sum")
        loss = reduce(loss_per_batch, "b -> ", "mean")

        return loss, acc

    def training_step(self, batch, batch_idx):
        """Pytorch lightning method."""
        images, tile_catalog, _ = parse_dataset(batch, tile_slen=self.tile_slen)
        loss, acc = self.get_loss(images, tile_catalog)
        self.log("train/loss", loss, batch_size=len(images))
        self.log("train/acc", acc, batch_size=len(images))
        return loss

    def validation_step(self, batch, batch_idx):
        """Pytorch lightning method."""
        images, tile_catalog, _ = parse_dataset(batch, tile_slen=self.tile_slen)
        loss, acc = self.get_loss(images, tile_catalog)
        self.log("val/loss", loss, batch_size=len(images))
        self.log("val/acc", acc, batch_size=len(images))
        return loss

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=1e-4)

    def _center_ptiles(self, image_ptiles: Tensor, tile_locs_flat: Tensor) -> Tensor:
        assert image_ptiles.shape[-1] == image_ptiles.shape[-2] == self.ptile_slen
        shifted_ptiles = shift_sources_bilinear(
            image_ptiles,
            tile_locs_flat,
            tile_slen=self.tile_slen,
            slen=self.ptile_slen,
            center=True,
        )
        assert shifted_ptiles.shape[-1] == shifted_ptiles.shape[-2] == self.ptile_slen
        cropped_ptiles = shifted_ptiles[
            ...,
            self.tile_slen : (self.ptile_slen - self.tile_slen),
            self.tile_slen : (self.ptile_slen - self.tile_slen),
        ]
        assert cropped_ptiles.shape[-1] == cropped_ptiles.shape[-2] == self.final_slen
        return cropped_ptiles
