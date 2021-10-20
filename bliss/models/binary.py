import pytorch_lightning as pl
import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.nn import BCELoss

from bliss.models.decoder import get_mgrid
from bliss.models.encoder import EncoderCNN, get_images_in_tiles, get_is_on_from_n_sources
from bliss.models.galaxy_encoder import center_ptiles, get_full_params
from bliss.optimizer import get_optimizer
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

    def center_ptiles(self, image_ptiles, tile_locs):
        """Return padded tiles centered on each corresponding source."""
        return center_ptiles(
            image_ptiles,
            tile_locs,
            self.tile_slen,
            self.ptile_slen,
            self.border_padding,
            self.swap,
            self.cached_grid,
        )

    def forward_image(self, images, tile_locs):
        """Splits `images` into padded tiles and runs `self.forward()` on them."""
        batch_size = images.shape[0]
        ptiles = self.get_images_in_tiles(images)
        prob_galaxy = self(ptiles, tile_locs)
        return prob_galaxy.view(batch_size, -1, 1, 1)

    def forward(self, image_ptiles, tile_locs):
        """Centers padded tiles using `tile_locs` and runs the binary encoder on them."""
        assert image_ptiles.shape[-1] == image_ptiles.shape[-2] == self.ptile_slen
        n_ptiles = image_ptiles.shape[0]

        # in each padded tile we need to center the corresponding galaxy/star
        tile_locs = tile_locs.reshape(n_ptiles, self.max_sources, 2)
        centered_ptiles = self.center_ptiles(image_ptiles, tile_locs)
        assert centered_ptiles.shape[-1] == centered_ptiles.shape[-2] == self.slen

        # forward to layer shared by all n_sources
        log_img = torch.log(centered_ptiles - centered_ptiles.min() + 1.0)
        h = self.enc_conv(log_img)
        z = self.enc_final(h).reshape(n_ptiles, 1)
        return torch.sigmoid(z).clamp(1e-4, 1 - 1e-4)

    def get_prediction(self, batch):
        """Return loss, accuracy, binary probabilities, and MAP classifications for given batch."""

        images = batch["images"]
        galaxy_bool = batch["galaxy_bool"].reshape(-1)
        prob_galaxy = self.forward_image(images, batch["locs"]).reshape(-1)
        tile_is_on_array = get_is_on_from_n_sources(batch["n_sources"], self.max_sources)
        tile_is_on_array = tile_is_on_array.reshape(-1)

        # we need to calculate cross entropy loss, only for "on" sources
        loss = BCELoss(reduction="none")(prob_galaxy, galaxy_bool) * tile_is_on_array
        loss = loss.sum()

        # get predictions for calculating metrics
        pred_galaxy_bool = (prob_galaxy > 0.5).float() * tile_is_on_array
        correct = ((pred_galaxy_bool.eq(galaxy_bool)) * tile_is_on_array).sum()
        total_n_sources = batch["n_sources"].sum()
        acc = correct / total_n_sources

        # finally organize quantities and return as a dictionary
        pred_star_bool = (1 - pred_galaxy_bool) * tile_is_on_array

        return {
            "loss": loss,
            "acc": acc,
            "galaxy_bool": pred_galaxy_bool.reshape(images.shape[0], -1, 1, 1),
            "star_bool": pred_star_bool.reshape(images.shape[0], -1, 1, 1),
            "prob_galaxy": prob_galaxy.reshape(images.shape[0], -1, 1, 1),
        }

    def configure_optimizers(self):
        """Pytorch lightning method."""
        assert self.hparams["optimizer_params"] is not None, "Need to specify 'optimizer_params'."
        name = self.hparams["optimizer_params"]["name"]
        kwargs = self.hparams["optimizer_params"]["kwargs"]
        return get_optimizer(name, self.parameters(), kwargs)

    def training_step(self, batch, batch_idx):
        """Pytorch lightning method."""
        pred = self.get_prediction(batch)
        self.log("train/loss", pred["loss"])
        self.log("train/acc", pred["acc"])
        return pred["loss"]

    def validation_step(self, batch, batch_idx):
        """Pytorch lightning method."""
        pred = self.get_prediction(batch)
        self.log("val/loss", pred["loss"])
        self.log("val/acc", pred["acc"])
        return batch

    def validation_epoch_end(self, outputs):
        """Pytorch lightning method."""
        # Put all outputs together into a single batch
        batch = {}
        for b in outputs:
            for k, v in b.items():
                curr_val = batch.get(k, torch.tensor([], device=v.device))
                batch[k] = torch.cat([curr_val, v])

        if self.n_bands == 1:
            self.make_plots(batch)

    def test_step(self, batch, batch_idx):
        """Pytorch lightning method."""
        pred = self.get_prediction(batch)
        self.log("acc", pred["acc"])

    def make_plots(self, batch, n_samples=16):
        """Produced informative plots demonstrating encoder performance."""

        assert n_samples ** (0.5) % 1 == 0
        if n_samples > len(batch["n_sources"]):  # do nothing if low on samples.
            return
        nrows = int(n_samples ** 0.5)  # for figure

        # extract non-params entries so that 'get_full_params' to works.
        exclude = {"images", "slen", "background"}
        slen = int(batch["slen"].unique().item())
        tile_params = {k: v for k, v in batch.items() if k not in exclude}
        true_params = get_full_params(tile_params, slen)

        # prediction
        pred = self.get_prediction(batch)
        tile_est = dict(tile_params.items())
        tile_est["galaxy_bool"] = pred["galaxy_bool"]
        tile_est["star_bool"] = pred["star_bool"]
        tile_est["prob_galaxy"] = pred["prob_galaxy"]
        est = get_full_params(tile_est, slen)

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
                prob_galaxy=est["prob_galaxy"],
            )

        fig.tight_layout()

        title = f"Epoch:{self.current_epoch}/Validation Images"
        if self.logger is not None:
            self.logger.experiment.add_figure(title, fig)
