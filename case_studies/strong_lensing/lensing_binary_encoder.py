# pylint: disable=R

from typing import Dict, Optional, Union

import pytorch_lightning as pl
import torch
from einops import rearrange
from matplotlib import pyplot as plt
from torch import Tensor
from torch.nn import BCELoss
from torch.optim import Adam

from bliss.catalog import TileCatalog, get_images_in_tiles, get_is_on_from_n_sources
from bliss.models.encoder_layers import (
    ConcatBackgroundTransform,
    EncoderCNN,
    LogBackgroundTransform,
    make_enc_final,
)
from bliss.models.galaxy_encoder import CenterPaddedTilesTransform
from bliss.plotting import add_loc_legend, plot_image, plot_locs


class LensingBinaryEncoder(pl.LightningModule):
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
        optimizer_params: Optional[dict] = None,
    ):
        """Encoder which conditioned on other source params returns probability of being lensed.

        This class implements the binary encoder, which is supposed to take in a synthetic image
        along with true locations and source parameters and return whether each source in that
        image is a lensed galaxy. Note that this is structurally identical to the BinaryEncoder
        used for distinguishing stars from galaxies (future refactor)

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
        self.center_ptiles = CenterPaddedTilesTransform(self.tile_slen, self.ptile_slen)

    def forward(self, image_ptiles, locs):
        return self.encode(image_ptiles, locs)

    def encode(self, image_ptiles: Tensor, locs: Tensor) -> Tensor:
        """Runs the binary encoder on centered_ptiles."""
        n_samples, n_ptiles, max_sources, _ = locs.shape
        assert max_sources == self.max_sources

        centered_tiles = self._get_images_in_centered_tiles(image_ptiles, locs)
        assert centered_tiles.shape[-1] == centered_tiles.shape[-2] == self.slen

        # forward to layer shared by all n_sources
        x = rearrange(centered_tiles, "ns np h c w -> (ns np) h c w")
        h = self.enc_conv(x)
        h2 = self.enc_final(h)
        lensed_galaxy_probs = torch.sigmoid(h2).clamp(1e-4, 1 - 1e-4)
        return rearrange(
            lensed_galaxy_probs,
            "(ns npt ms) 1 -> ns npt ms 1",
            ns=n_samples,
            npt=n_ptiles,
            ms=max_sources,
        )

    def get_prediction(self, batch: Dict[str, Tensor]):
        """Return loss, accuracy, binary probabilities, and MAP classifications for given batch."""

        lensed_galaxy_bools = batch["lensed_galaxy_bools"].reshape(-1)
        locs = rearrange(batch["locs"], "n nth ntw ms hw -> 1 (n nth ntw) ms hw")
        image_ptiles = get_images_in_tiles(
            torch.cat((batch["images"], batch["background"]), dim=1),
            self.tile_slen,
            self.ptile_slen,
        )
        image_ptiles = rearrange(image_ptiles, "n nth ntw b h w -> (n nth ntw) b h w")
        lensed_galaxy_probs = self.forward(image_ptiles, locs)
        lensed_galaxy_probs = lensed_galaxy_probs.reshape(-1)

        tile_is_on_array = get_is_on_from_n_sources(batch["n_sources"], self.max_sources)
        tile_is_on_array = tile_is_on_array.reshape(-1)

        # we need to calculate cross entropy loss, only for "on" sources
        loss = (
            BCELoss(reduction="none")(lensed_galaxy_probs, lensed_galaxy_bools) * tile_is_on_array
        )
        loss = loss.sum()

        # get predictions for calculating metrics
        pred_lensed_galaxy_bools = (lensed_galaxy_probs > 0.5).float() * tile_is_on_array
        correct = ((pred_lensed_galaxy_bools.eq(lensed_galaxy_bools)) * tile_is_on_array).sum()
        total_n_sources = batch["n_sources"].sum()
        acc = correct / total_n_sources

        return {
            "loss": loss,
            "acc": acc,
            "lensed_galaxy_bools": pred_lensed_galaxy_bools,
            "lensed_galaxy_probs": lensed_galaxy_probs,
        }

    def _get_images_in_centered_tiles(self, image_ptiles: Tensor, tile_locs: Tensor) -> Tensor:
        log_image_ptiles = self.input_transform(image_ptiles)
        assert log_image_ptiles.shape[-1] == log_image_ptiles.shape[-2] == self.ptile_slen
        # in each padded tile we need to center the corresponding galaxy/star
        return self.center_ptiles(log_image_ptiles, tile_locs)

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
        true_param_names = {"locs", "n_sources", "lensed_galaxy_bools"}
        true_tile_params = TileCatalog(
            self.tile_slen, {k: v for k, v in batch.items() if k in true_param_names}
        )
        true_params = true_tile_params.to_full_params()
        # prediction
        tile_est = true_tile_params.copy()
        shape = tile_est["lensed_galaxy_bools"].shape
        tile_est["lensed_galaxy_bools"] = batch["pred_lensed_galaxy_bools"].reshape(*shape)
        tile_est["lensed_galaxy_probs"] = batch["pred_lensed_galaxy_probs"].reshape(*shape)
        est = tile_est.to_full_params()
        # setup figure and axes
        fig, axes = plt.subplots(nrows=nrows, ncols=nrows, figsize=(12, 12))
        axes = axes.flatten()

        for ii in range(n_samples):
            ax = axes[ii]
            labels = None if ii > 0 else ("t. gal", "p. gal", "t. star", "p. star")
            bp = self.border_padding
            image = batch["images"][ii, 0].cpu().numpy()
            true_plocs = true_params.plocs[ii].cpu().numpy().reshape(-1, 2)
            true_gbools = true_params["lensed_galaxy_bools"][ii].cpu().numpy().reshape(-1)
            est_plocs = est.plocs[ii].cpu().numpy().reshape(-1, 2)
            est_gprobs = est["lensed_galaxy_probs"][ii].cpu().numpy().reshape(-1)
            slen, _ = image.shape
            plot_image(fig, ax, image, colorbar=False)
            plot_locs(ax, bp, slen, true_plocs, true_gbools, m="+", s=30, cmap="cool")
            plot_locs(ax, bp, slen, est_plocs, est_gprobs, m="x", s=20, cmap="bwr")
            if ii == 0:
                add_loc_legend(ax, labels)

        fig.tight_layout()

        title = f"Epoch:{self.current_epoch}/Validation_Images"
        if self.logger is not None:
            self.logger.experiment.add_figure(title, fig)
