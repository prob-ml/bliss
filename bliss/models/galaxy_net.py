import pytorch_lightning as pl
import matplotlib.pyplot as plt
from omegaconf import DictConfig

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import L1Loss, MSELoss

from bliss.optimizer import get_optimizer

plt.switch_backend("Agg")


class GalaxyEncoder(nn.Module):
    def __init__(self, slen=48, latent_dim=8, n_bands=1, hidden=256):
        super().__init__()

        self.slen = slen
        self.latent_dim = latent_dim
        self.n_bands = n_bands

        min_slen = slen / 2 ** 4  # e.g. slen = 64, min_slen = 4
        assert min_slen % 1 == 0
        min_slen = int(min_slen)

        self.features = nn.Sequential(
            nn.Conv2d(self.n_bands, 8, 3, stride=1, padding=1),  # e.g. input=64, 64x64
            nn.LeakyReLU(),
            nn.Conv2d(8, 8, 3, stride=2, padding=1),  # 32x32
            nn.LeakyReLU(),
            nn.Conv2d(8, 16, 3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(16, 16, 3, stride=2, padding=1),  # 16x16
            nn.LeakyReLU(),
            nn.Conv2d(16, 32, 3, stride=1),
            nn.LeakyReLU(),
            nn.Conv2d(32, 32, 3, stride=2, padding=1),  # 8x8
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, 3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, 3, stride=2, padding=1),  # 4x4
            nn.LeakyReLU(),
            nn.Flatten(1, -1),
            nn.Linear(64 * min_slen ** 2, hidden),
            nn.LeakyReLU(),
            nn.Linear(hidden, self.latent_dim),
        )

    def forward(self, image):
        return self.features(image)


class GalaxyDecoder(nn.Module):
    def __init__(self, slen=48, latent_dim=8, n_bands=1, hidden=256):
        super().__init__()

        self.slen = slen
        self.latent_dim = latent_dim
        self.n_bands = n_bands

        min_slen = slen / 2 ** 4  # e.g. slen = 64, min_slen = 4
        assert min_slen % 1 == 0
        self.min_slen = int(min_slen)

        self.fc = nn.Sequential(
            nn.Linear(latent_dim, hidden),
            nn.LeakyReLU(),
            nn.Linear(hidden, 64 * self.min_slen ** 2),
            nn.LeakyReLU(),
        )

        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(64, 64, 3, padding=1, stride=2, output_padding=1),  # 8x8
            nn.LeakyReLU(),
            nn.ConvTranspose2d(64, 32, 3, padding=1, stride=1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(32, 32, 3, padding=1, stride=2, output_padding=1),  # 16x16
            nn.LeakyReLU(),
            nn.ConvTranspose2d(32, 16, 3, padding=1, stride=1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(16, 16, 3, padding=1, stride=2, output_padding=1),  # 32x32
            nn.LeakyReLU(),
            nn.ConvTranspose2d(16, 8, 3, padding=1, stride=1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(8, 8, 3, padding=1, stride=2, output_padding=1),  # 64x64
            nn.LeakyReLU(),
            nn.ConvTranspose2d(8, self.n_bands, 3, padding=1, stride=1),  # 2x64x64
        )

    def forward(self, z):
        z = self.fc(z)
        z = z.view(-1, 64, self.min_slen, self.min_slen)
        z = self.deconv(z)
        z = z[:, :, : self.slen, : self.slen]
        assert z.shape[-1] == self.slen and z.shape[-2] == self.slen
        recon_mean = F.relu(z)
        return recon_mean


class OneCenteredGalaxy(pl.LightningModule):

    # ---------------
    # Model
    # ----------------

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters(cfg)

        self.enc = GalaxyEncoder(**cfg.model.params)
        self.dec = GalaxyDecoder(**cfg.model.params)

        self.warm_up = cfg.model.warm_up
        self.beta = cfg.model.beta
        self.loss_type = cfg.model.loss_type

        assert self.warm_up == 0 or self.beta == 1, "Only one of 'warm_up'/'beta' can be active."

        self.register_buffer("zero", torch.zeros(1))
        self.register_buffer("one", torch.ones(1))
        self.register_buffer("free_bits", torch.tensor([cfg.model.free_bits]))

    def forward(self, image, background):
        # sampling images from the real distribution
        # z | x ~ decoder

        # shape = [nsamples, latent_dim]
        z = self.enc.forward(image - background)

        # reconstructed mean/variances images (per pixel quantities)
        # no background
        recon_mean = self.dec.forward(z)
        recon_mean = recon_mean + background

        return recon_mean

    def get_loss(self, image, recon_mean):

        if self.loss_type == "L2":
            return MSELoss(reduction="sum")(image, recon_mean)

        if self.loss_type == "L1":
            return L1Loss(reduction="sum")(image, recon_mean)

        raise NotImplementedError("Loss not implemented.")

    # ---------------
    # Optimizer
    # ----------------

    def configure_optimizers(self):
        name = self.hparams.optimizer.name
        optim_params = self.hparams.optimizer.params
        return get_optimizer(name, self.parameters(), optim_params)

    # ---------------
    # Training
    # ----------------

    def training_step(self, batch, batch_idx):  # pylint: disable=unused-argument
        images, background = batch["images"], batch["background"]
        recon_mean = self(images, background)
        loss = self.get_loss(images, recon_mean)
        self.log("train_loss", loss)
        return loss

    # ---------------
    # Validation
    # ----------------

    def validation_step(self, batch, batch_idx):  # pylint: disable=unused-argument
        images, background = batch["images"], batch["background"]
        recon_mean = self(images, background)
        loss = self.get_loss(images, recon_mean)

        # metrics
        self.log("val_loss", loss)
        residuals = (images - recon_mean) / torch.sqrt(images)
        self.log("val_max_residual", residuals.abs().max())
        return {"images": images, "recon_mean": recon_mean}

    def validation_epoch_end(self, outputs):
        images = outputs[0]["images"][:10]
        recon_mean = outputs[0]["recon_mean"][:10]
        fig = self.plot_reconstruction(images, recon_mean)
        if self.logger:
            self.logger.experiment.add_figure(f"Images {self.current_epoch}", fig)

    def plot_reconstruction(self, images, recon_mean):
        # only plot i band if available, otherwise the highest band given.
        assert images.size(0) >= 10
        assert self.enc.n_bands == self.dec.n_bands
        n_bands = self.enc.n_bands
        num_examples = 10
        num_cols = 3
        band_idx = min(2, n_bands - 1)
        residuals = (images - recon_mean) / torch.sqrt(images)
        plt.ioff()

        fig = plt.figure(figsize=(10, 25))
        plt.suptitle("Epoch {:d}".format(self.current_epoch))

        for i in range(num_examples):

            plt.subplot(num_examples, num_cols, num_cols * i + 1)
            plt.title("images")
            plt.imshow(images[i, band_idx].data.cpu().numpy())
            plt.colorbar()

            plt.subplot(num_examples, num_cols, num_cols * i + 2)
            plt.title("recon_mean")
            plt.imshow(recon_mean[i, band_idx].data.cpu().numpy())
            plt.colorbar()

            plt.subplot(num_examples, num_cols, num_cols * i + 3)
            plt.title("residuals")
            plt.imshow(residuals[i, band_idx].data.cpu().numpy())
            plt.colorbar()

        plt.tight_layout()

        return fig

    def test_step(self, batch, batch_idx):  # pylint: disable=unused-argument
        images, background = batch["images"], batch["background"]
        recon_mean = self(images, background)
        residuals = (images - recon_mean) / torch.sqrt(images)
        self.log("max_residual", residuals.abs().max())
