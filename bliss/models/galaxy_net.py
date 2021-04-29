import pytorch_lightning as pl
import matplotlib.pyplot as plt
from omegaconf import DictConfig

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.optim import Adam
from torch.nn import L1Loss, MSELoss

plt.switch_backend("Agg")


class CenteredGalaxyEncoder(nn.Module):
    def __init__(self, slen=44, latent_dim=8, n_bands=1, hidden=256):
        super().__init__()

        f = lambda x: (x - 5) // 3 + 1  # function to figure out dimension of conv2d output.
        min_slen = f(f(slen))

        self.features = nn.Sequential(
            nn.Conv2d(n_bands, 4, 5, stride=3, padding=0),
            nn.LeakyReLU(),
            nn.Conv2d(4, 8, 5, stride=3, padding=0),
            nn.LeakyReLU(),
            nn.Flatten(1, -1),
            nn.Linear(8 * min_slen ** 2, hidden),
            nn.LeakyReLU(),
            nn.Linear(hidden, latent_dim),
        )

    def forward(self, image):
        return self.features(image)


class CenteredGalaxyDecoder(nn.Module):
    def __init__(self, slen=44, latent_dim=8, n_bands=1, hidden=256):
        super().__init__()

        self.slen = slen

        f = lambda x: (x - 5) // 3 + 1  # function to figure out dimension of conv2d output.
        g = lambda x: (x - 1) * 3 + 5
        self.min_slen = f(f(slen))
        assert g(g(self.min_slen)) == slen

        self.fc = nn.Sequential(
            nn.Linear(latent_dim, hidden),
            nn.LeakyReLU(),
            nn.Linear(hidden, 8 * self.min_slen ** 2),
            nn.LeakyReLU(),
        )

        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(8, 4, 5, stride=3),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(4, n_bands, 5, stride=3),
        )

    def forward(self, z):
        z = self.fc(z)
        z = z.view(-1, 8, self.min_slen, self.min_slen)
        z = self.deconv(z)
        z = z[:, :, : self.slen, : self.slen]
        assert z.shape[-1] == self.slen and z.shape[-2] == self.slen
        recon_mean = F.relu(z)
        return recon_mean


class OneCenteredGalaxyAE(pl.LightningModule):

    # ---------------
    # Model
    # ----------------

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters(cfg)

        self.enc = CenteredGalaxyEncoder(**cfg.model.params)
        self.dec = CenteredGalaxyDecoder(**cfg.model.params)

        self.loss_type = cfg.model.loss_type

        self.register_buffer("zero", torch.zeros(1))
        self.register_buffer("one", torch.ones(1))

    def forward(self, image, background):
        z = self.enc.forward(image - background)
        recon_mean = self.dec.forward(z)
        recon_mean = recon_mean + background
        return recon_mean

    def get_loss(self, image, recon_mean):

        if self.loss_type == "gauss":
            return -Normal(recon_mean, recon_mean.sqrt()).log_prob(image).sum()

        if self.loss_type == "L2":
            return MSELoss(reduction="sum")(image, recon_mean)

        if self.loss_type == "L1":
            return L1Loss(reduction="sum")(image, recon_mean)

        raise NotImplementedError("Loss not implemented.")

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
        num_examples = 10
        num_cols = 3
        residuals = (images - recon_mean) / torch.sqrt(images)
        plt.ioff()

        fig = plt.figure(figsize=(10, 25))
        plt.suptitle("Epoch {:d}".format(self.current_epoch))

        for i in range(num_examples):

            plt.subplot(num_examples, num_cols, num_cols * i + 1)
            plt.title("images")
            plt.imshow(images[i, 0].data.cpu().numpy())
            plt.colorbar()

            plt.subplot(num_examples, num_cols, num_cols * i + 2)
            plt.title("recon_mean")
            plt.imshow(recon_mean[i, 0].data.cpu().numpy())
            plt.colorbar()

            plt.subplot(num_examples, num_cols, num_cols * i + 3)
            plt.title("residuals")
            plt.imshow(residuals[i, 0].data.cpu().numpy())
            plt.colorbar()

        plt.tight_layout()

        return fig

    def test_step(self, batch, batch_idx):  # pylint: disable=unused-argument
        images, background = batch["images"], batch["background"]
        recon_mean = self(images, background)
        residuals = (images - recon_mean) / torch.sqrt(images)
        self.log("max_residual", residuals.abs().max())
