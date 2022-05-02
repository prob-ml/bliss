from typing import Dict, Union

import pytorch_lightning as pl
import torch
from einops import rearrange
from matplotlib import pyplot as plt
from torch import Tensor
from torch.nn import BCELoss
from torch.optim import Adam

from bliss.catalog import TileCatalog, get_images_in_tiles, get_is_on_from_n_sources
from bliss.models.decoder import get_mgrid
from bliss.models.galaxy_encoder import center_ptiles
from bliss.models.location_encoder import (
    ConcatBackgroundTransform,
    EncoderCNN,
    LogBackgroundTransform,
    make_enc_final,
)
from bliss.reporting import plot_image_and_locs


class BinaryEncoder(pl.LightningModule):
    def __init__(
        self,
        input_transform: Union[ConcatBackgroundTransform, LogBackgroundTransform],
        n_bands: int,
        tile_slen: int,
        ptile_slen: int,
        channel: int,
        hidden: int,
        spatial_dropout: float,
        dropout: float,
        optimizer_params: dict = None,
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
        self.optimizer_params = optimizer_params
        self.input_transform = input_transform

        self.max_sources = 1  # by construction.

        # extract useful info from image_decoder
        self.n_bands = n_bands

        # put image dimensions together
        self.tile_slen = tile_slen
        self.ptile_slen = ptile_slen
        border_padding = (ptile_slen - tile_slen) / 2
        assert tile_slen <= ptile_slen
        assert border_padding % 1 == 0, "amount of border padding should be an integer"
        self.border_padding = int(border_padding)
        self.slen = self.ptile_slen - 2 * self.tile_slen  # will always crop 2 * tile_slen

        dim_enc_conv_out = ((self.slen + 1) // 2 + 1) // 2
        n_bands_in = self.input_transform.output_channels(n_bands)
        self.enc_conv = EncoderCNN(n_bands_in, channel, spatial_dropout)
        self.enc_final = make_enc_final(channel * 4 * dim_enc_conv_out**2, hidden, 1, dropout)

        # grid for center cropped tiles
        self.register_buffer("cached_grid", get_mgrid(self.ptile_slen), persistent=False)
        self.register_buffer("swap", torch.tensor([1, 0]), persistent=False)

    def forward(self, image_ptiles, locs):
        return self.encode(image_ptiles, locs)

    def encode(self, image_ptiles: Tensor, locs: Tensor) -> Tensor:
        """Runs the binary encoder on centered_ptiles."""
        centered_tiles = self._get_images_in_centered_tiles(image_ptiles, locs)
        assert centered_tiles.shape[-1] == centered_tiles.shape[-2] == self.slen

        # forward to layer shared by all n_sources
        h = self.enc_conv(centered_tiles)
        h2 = self.enc_final(h)
        galaxy_probs = torch.sigmoid(h2).clamp(1e-4, 1 - 1e-4)
        return rearrange(galaxy_probs, "(npt ns) 1 -> npt ns 1", ns=self.max_sources)

    def get_prediction(self, batch: Dict[str, Tensor]):
        """Return loss, accuracy, binary probabilities, and MAP classifications for given batch."""

        galaxy_bools = batch["galaxy_bools"].reshape(-1)
        locs = rearrange(batch["locs"], "n nth ntw ns hw -> (n nth ntw) ns hw")
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
        return center_ptiles(
            log_image_ptiles,
            tile_locs,
            self.tile_slen,
            self.ptile_slen,
            self.border_padding,
            self.swap,
            self.cached_grid,
        )

    def configure_optimizers(self):
        """Pytorch lightning method."""
        return Adam(self.parameters(), **self.optimizer_params)

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
        return {**batch, **pred_out}

    def validation_epoch_end(self, outputs):
        """Pytorch lightning method."""
        # Put all outputs together into a single batch
        batch = {}
        for b in outputs:
            for k, v in b.items():
                curr_val = batch.get(k, torch.tensor([], device=v.device))
                if not v.shape:
                    v = v.reshape(1)
                batch[k] = torch.cat((curr_val, v))

        if self.n_bands == 1:
            self.make_plots(batch)

    def test_step(self, batch, batch_idx):
        """Pytorch lightning method."""
        batch_size = len(batch["images"])
        pred = self.get_prediction(batch)
        self.log("acc", pred["acc"], batch_size=batch_size)

    def make_plots(self, batch, n_samples=16):
        """Produced informative plots demonstrating encoder performance."""

        assert n_samples ** (0.5) % 1 == 0
        if n_samples > len(batch["n_sources"]):  # do nothing if low on samples.
            return
        nrows = int(n_samples**0.5)  # for figure

        # extract non-params entries so that 'get_full_params' to works.
        true_param_names = {"locs", "n_sources", "galaxy_bools", "star_bools"}
        true_tile_params = TileCatalog(
            self.tile_slen, {k: v for k, v in batch.items() if k in true_param_names}
        )
        true_params = true_tile_params.to_full_params()
        # prediction
        tile_est = true_tile_params.copy()
        shape = tile_est["galaxy_bools"].shape
        tile_est["galaxy_bools"] = batch["pred_galaxy_bools"].reshape(*shape)
        tile_est["star_bools"] = batch["pred_star_bools"].reshape(*shape)
        tile_est["galaxy_probs"] = batch["pred_galaxy_probs"].reshape(*shape)
        est = tile_est.to_full_params()
        # setup figure and axes
        fig, axes = plt.subplots(nrows=nrows, ncols=nrows, figsize=(12, 12))
        axes = axes.flatten()

        for i in range(n_samples):
            labels = None if i > 0 else ("t. gal", "p. gal", "t. star", "p. star")
            bp = self.border_padding
            images = batch["images"]
            plot_image_and_locs(
                fig, axes[i], i, images, bp, true_params, est, labels=labels, annotate_probs=True
            )

        fig.tight_layout()

        title = f"Epoch:{self.current_epoch}/Validation Images"
        if self.logger is not None:
            self.logger.experiment.add_figure(title, fig)
