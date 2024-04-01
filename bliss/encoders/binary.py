from typing import Dict, List, Union

import torch
from einops import rearrange
from torch import Tensor, nn
from torch.nn import BCELoss

from bliss.catalog import get_images_in_tiles, get_is_on_from_n_sources
from bliss.encoders.deblend import CenterPaddedTilesTransform
from bliss.encoders.layers import (
    ConcatBackgroundTransform,
    EncoderCNN,
    LogBackgroundTransform,
    make_enc_final,
)


class BinaryEncoder(nn.Module):
    def __init__(
        self,
        input_transform: Union[ConcatBackgroundTransform, LogBackgroundTransform],
        n_bands: int = 1,
        tile_slen: int = 4,
        ptile_slen: int = 52,
        channel: int = 8,
        hidden: int = 128,
        spatial_dropout: float = 0,
        dropout: float = 0,
    ):
        """Encoder which conditioned on other source params returns probability of galaxy vs. star.

        This class implements the binary encoder, which is supposed to take in a synthetic image
        along with true locations and source parameters and return whether each source in that
        image is a star or a galaxy.

        Arguments:
            input_transform: Transformation to apply to input image.
            n_bands: number of bands
            tile_slen: dimension (in pixels) of each tile.
            ptile_slen: dimension (in pixels) of the individual image padded tiles.
            channel: TODO (document this)
            hidden: TODO (document this)
            spatial_dropout: TODO (document this)
            dropout: TODO (document this)
            optimizer_params: TODO (document this)
        """
        super().__init__()
        self.save_hyperparameters()
        self.input_transform = input_transform

        self.max_sources = 1  # by construction.

        # extract useful info from image_decoder
        self.n_bands = n_bands

        # put image dimensions together
        self.tile_slen = tile_slen
        self.ptile_slen = ptile_slen
        bp = (ptile_slen - tile_slen) / 2
        assert tile_slen <= ptile_slen
        assert bp.is_integer(), "amount of border padding should be an integer"
        self.bp = int(bp)
        self.slen = self.ptile_slen - 2 * self.tile_slen  # will always crop 2 * tile_slen

        dim_enc_conv_out = ((self.slen + 1) // 2 + 1) // 2
        n_bands_in = self.input_transform.output_channels(n_bands)
        self.enc_conv = EncoderCNN(n_bands_in, channel, spatial_dropout)
        self.enc_final = make_enc_final(channel * 4 * dim_enc_conv_out**2, hidden, 1, dropout)

        # grid for center cropped tiles
        self.center_ptiles = CenterPaddedTilesTransform(self.tile_slen, self.ptile_slen)

        self.validation_step_outputs: List[dict] = []

    def forward(self, image_ptiles: Tensor, locs: Tensor) -> Tensor:
        """Runs the binary encoder on centered_ptiles."""
        n_samples, n_ptiles, max_sources, _ = locs.shape
        assert max_sources == self.max_sources

        centered_tiles = self._get_images_in_centered_tiles(image_ptiles, locs)
        assert centered_tiles.shape[-1] == centered_tiles.shape[-2] == self.slen

        # forward to layer shared by all n_sources
        x = rearrange(centered_tiles, "ns np h c w -> (ns np) h c w")
        h = self.enc_conv(x)
        h2 = self.enc_final(h)
        galaxy_probs = torch.sigmoid(h2).clamp(1e-4, 1 - 1e-4)
        return rearrange(
            galaxy_probs, "(ns npt ms) 1 -> ns npt ms 1", ns=n_samples, npt=n_ptiles, ms=max_sources
        )

    def get_prediction(self, batch: Dict[str, Tensor]):
        """Return loss, accuracy, binary probabilities, and MAP classifications for given batch."""

        galaxy_bools = batch["galaxy_bools"].reshape(-1)
        locs = rearrange(batch["locs"], "n nth ntw ms hw -> 1 (n nth ntw) ms hw")
        image_ptiles = get_images_in_tiles(
            torch.cat((batch["images"], batch["background"]), dim=1),
            self.tile_slen,
            self.ptile_slen,
        )
        image_ptiles = rearrange(image_ptiles, "n nth ntw b h w -> (n nth ntw) b h w")
        galaxy_probs = self.forward(image_ptiles, locs)
        galaxy_probs = galaxy_probs.reshape(-1)

        tile_is_on_array = get_is_on_from_n_sources(batch["n_sources"], self.max_sources)
        tile_is_on_array = tile_is_on_array.reshape(-1)

        # we need to calculate cross entropy loss, only for "on" sources
        loss = BCELoss(reduction="none")(galaxy_probs, galaxy_bools) * tile_is_on_array
        loss = loss.sum()

        # get predictions for calculating metrics
        pred_galaxy_bools = (galaxy_probs > 0.5).float() * tile_is_on_array
        correct = ((pred_galaxy_bools.eq(galaxy_bools)) * tile_is_on_array).sum()
        total_n_sources = batch["n_sources"].sum()
        acc = correct / total_n_sources

        # finally organize quantities and return as a dictionary
        pred_star_bools = (1 - pred_galaxy_bools) * tile_is_on_array

        return {
            "loss": loss,
            "acc": acc,
            "galaxy_bools": pred_galaxy_bools,
            "star_bools": pred_star_bools,
            "galaxy_probs": galaxy_probs,
        }

    def _get_images_in_centered_tiles(self, image_ptiles: Tensor, tile_locs: Tensor) -> Tensor:
        log_image_ptiles = self.input_transform(image_ptiles)
        assert log_image_ptiles.shape[-1] == log_image_ptiles.shape[-2] == self.ptile_slen
        # in each padded tile we need to center the corresponding galaxy/star
        return self.center_ptiles(log_image_ptiles, tile_locs)

    def training_step(self, batch, batch_idx):
        """Pytorch lightning method."""
        batch_size = len(batch["images"])
        pred = self.get_prediction(batch)
        self.log("train/loss", pred["loss"], batch_size=batch_size)
        self.log("train/acc", pred["acc"], batch_size=batch_size)
        return pred["loss"]

    def validation_step(self, batch, batch_idx):
        """Pytorch lightning method."""
        batch_size = len(batch["images"])
        pred = self.get_prediction(batch)
        self.log("val/loss", pred["loss"], batch_size=batch_size)
        self.log("val/acc", pred["acc"], batch_size=batch_size)
        pred_out = {f"pred_{k}": v for k, v in pred.items()}
        self.validation_step_outputs.append({**batch, **pred_out})
        return pred_out

    def on_validation_epoch_end(self):
        """Pytorch lightning method."""
        # Put all outputs together into a single batch
        batch = {}
        for b in self.validation_step_outputs:
            for k, v in b.items():
                curr_val = batch.get(k, torch.tensor([], device=v.device))
                if not v.shape:
                    v = v.reshape(1)
                batch[k] = torch.cat((curr_val, v))

        if self.n_bands == 1:
            self.make_plots(batch)

        self.validation_step_outputs.clear()
