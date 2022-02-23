import pytorch_lightning as pl
import torch
from einops import rearrange
from matplotlib import pyplot as plt
from torch import nn, Tensor
from torch.nn import BCELoss
from torch.optim import Adam

from bliss.models.decoder import get_mgrid
from bliss.models.galaxy_encoder import center_ptiles
from bliss.models.location_encoder import (
    EncoderCNN,
    get_full_params_from_tiles,
    get_images_in_tiles,
    get_is_on_from_n_sources,
    subtract_bg_and_log_transform,
)
from bliss.reporting import plot_image_and_locs


class BinaryEncoder(pl.LightningModule):
    def __init__(
        self,
        n_bands: int = 1,
        tile_slen: int = 4,
        ptile_slen: int = 52,
        channel: int = 8,
        hidden: int = 128,
        spatial_dropout: float = 0,
        dropout: float = 0,
        optimizer_params: dict = None,
    ):
        """Encoder which conditioned on other source params returns probability of galaxy vs. star.

        This class implements the binary encoder, which is supposed to take in a synthetic image
        along with true locations and source parameters and return whether each source in that
        image is a star or a galaxy.

        Arguments:
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
        self.enc_conv = EncoderCNN(self.n_bands, channel, spatial_dropout)
        self.enc_final = nn.Sequential(
            nn.Flatten(1),
            nn.Linear(channel * 4 * dim_enc_conv_out ** 2, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

        # grid for center cropped tiles
        self.register_buffer("cached_grid", get_mgrid(self.ptile_slen), persistent=False)
        self.register_buffer("swap", torch.tensor([1, 0]), persistent=False)

    def get_images_in_tiles(self, images):
        """Divide a batch of full images into padded tiles similar to nn.conv2d."""
        return get_images_in_tiles(images, self.tile_slen, self.ptile_slen)

    def flatten_and_center_ptiles(self, log_image_ptiles, tile_locs):
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

    def forward(self, log_image_ptiles: Tensor, tile_locs: Tensor):
        """Centers padded tiles using `tile_locs` and runs the binary encoder on them."""
        assert log_image_ptiles.shape[-1] == log_image_ptiles.shape[-2] == self.ptile_slen
        batch_size, n_tiles_h, n_tiles_w = tile_locs.shape[:3]

        # in each padded tile we need to center the corresponding galaxy/star
        centered_ptiles = self.flatten_and_center_ptiles(log_image_ptiles, tile_locs)
        assert centered_ptiles.shape[-1] == centered_ptiles.shape[-2] == self.slen

        # forward to layer shared by all n_sources
        h = self.enc_conv(centered_ptiles)
        h2 = self.enc_final(h)
        z = torch.sigmoid(h2).clamp(1e-4, 1 - 1e-4)
        return rearrange(
            z,
            "(b nth ntw s) 1 -> b nth ntw s 1",
            b=batch_size,
            nth=n_tiles_h,
            ntw=n_tiles_w,
            s=self.max_sources,
        )

    def get_prediction(self, batch):
        """Return loss, accuracy, binary probabilities, and MAP classifications for given batch."""

        images = batch["images"]
        background = batch["background"]
        galaxy_bools = batch["galaxy_bools"].reshape(-1)
        locs = batch["locs"]
        batch_size, n_tiles_h, n_tiles_w, max_sources, _ = locs.shape
        log_images = subtract_bg_and_log_transform(images, background)
        log_ptiles = self.get_images_in_tiles(log_images)
        galaxy_probs = self.forward(log_ptiles, locs).reshape(-1)
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

        ret = {
            "loss": loss,
            "acc": acc,
            "galaxy_bools": pred_galaxy_bools,
            "star_bools": pred_star_bools,
            "galaxy_probs": galaxy_probs,
        }

        for k in ("galaxy_bools", "star_bools", "galaxy_probs"):
            ret[k] = rearrange(
                ret[k],
                "(b nth ntw s) -> b nth ntw s 1",
                b=batch_size,
                nth=n_tiles_h,
                ntw=n_tiles_w,
                s=max_sources,
            )

        return ret

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
        return batch

    def validation_epoch_end(self, outputs):
        """Pytorch lightning method."""
        # Put all outputs together into a single batch
        batch = {}
        for b in outputs:
            for k, v in b.items():
                if k in {"hlen", "wlen"} and k not in batch:
                    batch[k] = v
                else:
                    curr_val = batch.get(k, torch.tensor([], device=v.device))
                    batch[k] = torch.cat([curr_val, v])

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
        nrows = int(n_samples ** 0.5)  # for figure

        # extract non-params entries so that 'get_full_params' to works.
        exclude = {"images", "background", "hlen", "wlen"}
        hlen = batch["hlen"].item()
        wlen = batch["wlen"].item()
        assert hlen == wlen
        slen = hlen
        true_tile_params = {k: v for k, v in batch.items() if k not in exclude}
        true_params = get_full_params_from_tiles(true_tile_params, self.tile_slen)
        # prediction
        pred = self.get_prediction(batch)
        tile_est = dict(true_tile_params.items())
        tile_est["galaxy_bools"] = pred["galaxy_bools"]
        tile_est["star_bools"] = pred["star_bools"]
        tile_est["galaxy_probs"] = pred["galaxy_probs"]
        est = get_full_params_from_tiles(tile_est, self.tile_slen)
        # setup figure and axes
        fig, axes = plt.subplots(nrows=nrows, ncols=nrows, figsize=(12, 12))
        axes = axes.flatten()

        for i in range(n_samples):
            plot_image_and_locs(
                i,
                fig,
                axes[i],
                batch["images"],
                slen,
                true_params,
                estimate=est,
                labels=None if i > 0 else ("t. gal", "p. gal", "t. star", "p. star"),
                annotate_axis=False,
                add_borders=True,
                galaxy_probs=est["galaxy_probs"],
            )

        fig.tight_layout()

        title = f"Epoch:{self.current_epoch}/Validation Images"
        if self.logger is not None:
            self.logger.experiment.add_figure(title, fig)
