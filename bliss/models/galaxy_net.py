import pytorch_lightning as pl
import matplotlib.pyplot as plt
from omegaconf import DictConfig

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.optim import Adam

plt.switch_backend("Agg")


class CenteredGalaxyEncoder(nn.Module):
    def __init__(self, slen=64, latent_dim=8, n_bands=1, hidden=256):
        # assert slen % 1 == 0

        super().__init__()

        self.slen = slen
        self.latent_dim = latent_dim
        self.n_bands = n_bands

        self.features = nn.Sequential(
            nn.BatchNorm2d(self.n_bands, momentum=0.5),
            nn.Conv2d(self.n_bands, 32, 3, stride=1, padding=1),  # e.g. input=64, 64x64
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=2, padding=1),  # 32x32
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=2, padding=1),  # 16x16
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, stride=2, padding=1),  # 8x8
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, stride=2, padding=1),  # 4x4
            nn.ReLU(),
            nn.Flatten(1, -1),
            nn.Linear(256 * 4 ** 2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )

        self.fc_mean = nn.Linear(hidden, self.latent_dim)
        self.fc_var = nn.Linear(hidden, self.latent_dim)

    def forward(self, subimage):
        z = self.features(subimage)
        z_mean = self.fc_mean(z)
        z_var = 1e-4 + torch.exp(self.fc_var(z))
        return z_mean, z_var


class CenteredGalaxyDecoder(nn.Module):
    def __init__(self, slen=41, latent_dim=8, n_bands=1, hidden=256):
        super().__init__()
        # assert slen % 1 == 0

        self.slen = slen
        self.latent_dim = latent_dim
        self.n_bands = n_bands

        self.fc = nn.Sequential(
            nn.Linear(latent_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 256 * 4 ** 2),
            nn.ReLU(),
        )

        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(256, 256, 3, padding=1, stride=2, output_padding=1),  # 8x8
            nn.ReLU(),
            nn.ConvTranspose2d(256, 256, 3, padding=1, stride=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 3, padding=1, stride=2, output_padding=1),  # 16x16
            nn.ReLU(),
            nn.ConvTranspose2d(128, 128, 3, padding=1, stride=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 3, padding=1, stride=2, output_padding=2),  # 32x32
            nn.ReLU(),
            nn.ConvTranspose2d(64, 64, 3, padding=1, stride=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, padding=1, stride=2, output_padding=1),  # 64x64
            nn.ConvTranspose2d(32, 2 * self.n_bands, 3, padding=1, stride=1),  # 6x64x64
        )

    def forward(self, z):
        z = self.fc(z)
        z = z.view(-1, 256, 4, 4)
        z = self.deconv(z)
        assert z.shape[-1] == self.slen and z.shape[-2] == self.slen
        z = z[:, :, : self.slen, : self.slen]
        recon_mean = F.relu(z[:, : self.n_bands])
        var_multiplier = 1 + 10 * torch.sigmoid(z[:, self.n_bands : (2 * self.n_bands)])
        recon_var = 1e-4 + var_multiplier * recon_mean
        return recon_mean, recon_var


class OneCenteredGalaxy(pl.LightningModule):

    # ---------------
    # Model
    # ----------------

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters(cfg)

        self.enc = CenteredGalaxyEncoder(**cfg.model.params)
        self.dec = CenteredGalaxyDecoder(**cfg.model.params)

        self.warm_up = cfg.model.warm_up

        self.register_buffer("zero", torch.zeros(1))
        self.register_buffer("one", torch.ones(1))

    def forward(self, image, background):
        # sampling images from the real distribution
        # z | x ~ decoder

        # shape = [nsamples, latent_dim]
        z_mean, z_var = self.enc.forward(image - background)

        q_z = Normal(z_mean, z_var.sqrt())
        z = q_z.rsample()

        log_q_z = q_z.log_prob(z).sum(1)
        p_z = Normal(self.zero, self.one)  # prior on z.
        log_p_z = p_z.log_prob(z).sum(1)
        kl_z = log_q_z - log_p_z  # log q(z | x) - log p(z)

        # reconstructed mean/variances images (per pixel quantities)
        # no background
        recon_mean, recon_var = self.dec.forward(z)

        # kl can behave wildly w/out background.
        recon_mean = recon_mean + background
        recon_var = recon_var + background

        return recon_mean, recon_var, kl_z

    def get_loss(self, image, recon_mean, recon_var, kl_z):
        # use a "deterministic warm-up" scheme to see if we get better results
        # on realistic galaxies. It involves "turning on" the prior penalty term slowly.
        # see: https://arxiv.org/pdf/1602.02282.pdf
        pwr = max(min(-self.warm_up + self.current_epoch, 0), -6)
        pr_penalty = kl_z * 10 ** pwr

        # return ELBO
        # NOTE: image includes background.
        # Covariance is diagonal in latent variables.
        # recon_loss = -log p(x | z), shape: torch.Size([ nsamples, n_bands, slen, slen])
        recon_losses = -Normal(recon_mean, recon_var.sqrt()).log_prob(image)
        recon_losses = recon_losses.view(image.size(0), -1).sum(1)
        loss = (recon_losses + pr_penalty).sum()

        return loss

    # ---------------
    # Optimizer
    # ----------------

    def configure_optimizers(self):
        params = self.hparams.optimizer.params
        return Adam(self.parameters(), **params)

    # ---------------
    # Training
    # ----------------

    def training_step(self, batch, batch_idx):  # pylint: disable=unused-argument
        images, background = batch["images"], batch["background"]
        recon_mean, recon_var, kl_z = self(images, background)
        loss = self.get_loss(images, recon_mean, recon_var, kl_z)
        self.log("train_loss", loss)
        return loss

    # ---------------
    # Validation
    # ----------------

    def validation_step(self, batch, batch_idx):  # pylint: disable=unused-argument
        images, background = batch["images"], batch["background"]
        recon_mean, recon_var, kl_z = self(images, background)
        loss = self.get_loss(images, recon_mean, recon_var, kl_z)

        # metrics
        self.log("val_loss", loss)
        residuals = (images - recon_mean) / torch.sqrt(images)
        self.log("val_max_residual", residuals.abs().max())
        return {"images": images, "recon_mean": recon_mean, "recon_var": recon_var}

    def validation_epoch_end(self, outputs):
        images = outputs[0]["images"][:10]
        recon_mean = outputs[0]["recon_mean"][:10]
        recon_var = outputs[0]["recon_var"][:10]
        fig = self.plot_reconstruction(images, recon_mean, recon_var)
        if self.logger:
            self.logger.experiment.add_figure(f"Images {self.current_epoch}", fig)

    def plot_reconstruction(self, images, recon_mean, recon_var):
        # only plot i band if available, otherwise the highest band given.
        assert images.size(0) >= 10
        assert self.enc.n_bands == self.dec.n_bands
        n_bands = self.enc.n_bands
        num_examples = 10
        num_cols = 4
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
            plt.title("recon_var")
            plt.imshow(recon_var[i, band_idx].data.cpu().numpy())
            plt.colorbar()

            plt.subplot(num_examples, num_cols, num_cols * i + 4)
            plt.title("residuals")
            plt.imshow(residuals[i, band_idx].data.cpu().numpy())
            plt.colorbar()

        plt.tight_layout()

        return fig

    def test_step(self, batch, batch_idx):  # pylint: disable=unused-argument
        images, background = batch["images"], batch["background"]
        recon_mean, _, _ = self(images, background)
        residuals = (images - recon_mean) / torch.sqrt(images)
        self.log("max_residual", residuals.abs().max())
