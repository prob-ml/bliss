import pytorch_lightning as pl
import torch
from matplotlib import pyplot as plt
from plotting import plot_image_and_locs
from torch import nn

from bliss.models.decoder import ImageDecoder, get_mgrid
from bliss.models.encoder import EncoderCNN, get_images_in_tiles, get_is_on_from_n_sources
from bliss.models.galaxy_encoder import center_ptiles, get_full_params


class BinaryEncoder(pl.LightningModule):
    def __init__(
        self,
        channel=8,
        spatial_dropout=0,
        dropout=0,
        hidden=128,
        decoder_kwargs: dict = None,
        optimizer_params: dict = None,  # pylint: disable=unused-argument
    ):
        """
        This class implements the binary encoder, which is supposed to take in a synthetic image
        along with true locations and source parameters and return whether each source in that
        image is a star or a galaxy.

        Args:
        slen (int): dimension of full image, we assume its square for now
        ptile_slen (int): dimension (in pixels) of the individual
                           image padded tiles (usually 8 for stars, and _ for galaxies).
        n_bands (int): number of bands
        max_detections (int): Number of maximum detections in a single tile.
        n_galaxy_params (int): Number of latent dimensions in the galaxy AE network.

        """
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
        self.log_softmax = nn.LogSoftmax(dim=1)

        # grid for center cropped tiles
        self.register_buffer("cached_grid", get_mgrid(self.ptile_slen), persistent=False)
        self.register_buffer("swap", torch.tensor([1, 0]), persistent=False)

    def get_images_in_tiles(self, images):
        """
        Divide a batch of full images into padded tiles similar to nn.conv2d
        with a sliding stride=self.tile_slen and window=self.ptile_slen
        """
        return get_images_in_tiles(images, self.tile_slen, self.ptile_slen)

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

    def forward_image(self, images, tile_locs):
        batch_size = images.shape[0]
        ptiles = self.get_images_in_tiles(images)
        prob_galaxy = self(ptiles, tile_locs)
        return prob_galaxy.view(batch_size, -1, 1, 1)

    def forward(self, image_ptiles, tile_locs):  # pylint: disable=empty-docstring
        """"""
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

        images = batch["images"]
        galaxy_bool = batch["galaxy_bool"]
        prob_galaxy = self.forward_image(images, batch["locs"])

        # we need to calculate cross entropy loss, only for "on" sources
        tile_is_on_array = get_is_on_from_n_sources(batch["n_sources"], self.max_sources)
        loss = (
            galaxy_bool * torch.log(prob_galaxy) + (1 - galaxy_bool) * torch.log(1 - prob_galaxy)
        ) * tile_is_on_array

        # get predictions for calculating metrics
        pred_galaxy_bool = (prob_galaxy > 0.5).astype(torch.float) * tile_is_on_array
        correct = ((pred_galaxy_bool.eq(galaxy_bool)) * tile_is_on_array).sum()
        total_n_sources = batch["n_sources"].sum()
        acc = correct / total_n_sources
        return {
            "loss": loss.sum(),
            "acc": acc,
            "galaxy_bool": pred_galaxy_bool,
            "prob_galaxy": prob_galaxy,
        }

    def training_step(self, batch, batch_idx):  # pylint: disable=unused-argument,empty-docstring
        """"""
        pred = self.get_loss(batch)
        self.log("train/loss", pred["loss"])
        self.log("val/acc", pred["acc"])
        return pred["loss"]

    def validation_step(self, batch, batch_idx):  # pylint: disable=unused-argument,empty-docstring
        """"""
        pred = self.get_loss(batch)
        self.log("val/loss", pred["loss"])
        self.log("val/acc", pred["acc"])
        return batch

    def validation_epoch_end(self, outputs):  # pylint: disable=empty-docstring
        """"""
        # put all outputs together into a single batch
        batch = {}
        for b in outputs:
            for k, v in b.items():
                curr_val = batch.get(k, torch.tensor([], device=v.device))
                batch[k] = torch.cat([curr_val, v])
        self.make_plots(batch)

    def make_plots(self, batch, n_samples=16):

        assert n_samples ** (0.5) % 1 == 0
        if n_samples > len(batch["n_sources"]):  # do nothing if low on samples.
            return

        # extract non-params entries so that 'get_full_params' to works.
        exclude = {"images", "slen", "background"}
        slen = int(batch["slen"].unique().item())
        tile_params = {k: v for k, v in batch.items() if k not in exclude}
        true_params = get_full_params(tile_params, slen)

        # prediction
        pred = self.get_prediction(batch)
        tile_est = {
            k: (v if k != "galaxy_bool" else pred["galaxy_bool"]) for k, v in tile_params.items()
        }
        tile_est["prob_galaxy"] = pred["prob_galaxy"]
        est = get_full_params(tile_est, slen)

        # setup figure and axes
        figsize = (12, 12)
        nrows = int(n_samples ** 0.5)
        fig, axes = plt.subplots(nrows=nrows, ncols=nrows, figsize=figsize)
        axes = axes.flatten()
        labels = ("t. gal", "p. gal", "t. star", "p. star")

        for i in range(n_samples):
            plot_image_and_locs(
                i,
                fig,
                axes[i],
                batch["images"],
                slen,
                true_params,
                estimate=est,
                labels=None if i > 0 else labels,
                annotate_axis=False,
                annotate_probs=self.annotate_probs,
                add_borders=True,
            )

        fig.tight_layout()
