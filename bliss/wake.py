import torch
from torch import optim
from torch.utils.data import DataLoader
from torch.distributions.normal import Normal
import pytorch_lightning as pl


class WakeNet(pl.LightningModule):

    # ---------------
    # Model
    # ----------------

    def __init__(
        self,
        star_encoder,
        image_decoder,
        observed_img,
        hparams,
    ):
        super(WakeNet, self).__init__()

        self.star_encoder = star_encoder
        self.image_decoder = image_decoder
        self.image_decoder.requires_grad_(True)
        assert self.image_decoder.galaxy_decoder is None

        self.slen = image_decoder.slen
        self.border_padding = image_decoder.border_padding

        # observed image is batch_size (or 1) x n_bands x slen x slen
        self.padded_slen = self.slen + 2 * self.border_padding
        assert len(observed_img.shape) == 4
        assert observed_img.shape[-1] == self.padded_slen, "cached grid won't match."

        self.observed_img = observed_img

        # hyper-parameters
        self.save_hyperparameters(hparams)
        self.n_samples = self.hparams["n_samples"]
        self.lr = self.hparams["lr"]

        # get n_bands
        self.n_bands = self.image_decoder.n_bands

    def forward(self, obs_img):

        with torch.no_grad():
            self.star_encoder.eval()
            sample = self.star_encoder.sample_encoder(obs_img, self.n_samples)

        shape = sample["locs"].shape[:-1]
        zero_gal_params = torch.zeros(*shape, self.image_decoder.n_galaxy_params)
        recon_mean, _ = self.image_decoder.render_images(
            sample["n_sources"].contiguous(),
            sample["locs"].contiguous(),
            sample["galaxy_bool"].contiguous(),
            zero_gal_params,
            sample["fluxes"].contiguous(),
            add_noise=False,
        )

        return recon_mean

    # ---------------
    # Data
    # ----------------

    def train_dataloader(self):
        return DataLoader(self.observed_img, batch_size=None)

    def val_dataloader(self):
        return DataLoader(self.observed_img, batch_size=None)

    # ---------------
    # Optimizer
    # ----------------

    def configure_optimizers(self):
        return optim.Adam([{"params": self.image_decoder.parameters(), "lr": self.lr}])

    # ---------------
    # Training
    # ----------------

    def get_loss(self, batch):
        img = batch.unsqueeze(0)
        recon_mean = self.forward(img)
        error = -Normal(recon_mean, recon_mean.sqrt()).log_prob(img)

        image_indx_start = self.border_padding
        image_indx_end = self.border_padding + self.slen
        loss = (
            error[
                :, :, image_indx_start:image_indx_end, image_indx_start:image_indx_end
            ]
            .sum((1, 2, 3))
            .mean()
        )
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.get_loss(batch)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.get_loss(batch)
        self.log("validation_loss", loss)
