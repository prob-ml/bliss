import pytorch_lightning as pl
import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.distributions import Normal
from torch.nn import functional as F

from bliss.optimizer import get_optimizer

plt.switch_backend("Agg")


class CenteredGalaxyEncoder(nn.Module):
    def __init__(self, slen=53, latent_dim=8, n_bands=1, hidden=256):
        super().__init__()

        self.slen = slen
        self.latent_dim = latent_dim

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

    def forward(self, image):  # pylint: disable=empty-docstring
        """"""
        return self.features(image)


class CenteredGalaxyDecoder(nn.Module):
    def __init__(self, slen=53, latent_dim=8, n_bands=1, hidden=256):
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

    def forward(self, z):  # pylint: disable=empty-docstring
        """"""
        z = self.fc(z)
        z = z.view(-1, 8, self.min_slen, self.min_slen)
        z = self.deconv(z)
        z = z[:, :, : self.slen, : self.slen]
        assert z.shape[-1] == self.slen and z.shape[-2] == self.slen
        return F.relu(z)


class OneCenteredGalaxyAE(pl.LightningModule):

    # ---------------
    # Model
    # ----------------

    def __init__(
        self,
        slen=53,
        latent_dim=8,
        hidden=256,
        n_bands=1,
        optimizer_params: dict = None,  # pylint: disable=unused-argument
    ):
        super().__init__()
        self.save_hyperparameters()

        self.enc = CenteredGalaxyEncoder(
            slen=slen, latent_dim=latent_dim, hidden=hidden, n_bands=n_bands
        )
        self.dec = CenteredGalaxyDecoder(
            slen=slen, latent_dim=latent_dim, hidden=hidden, n_bands=n_bands
        )

        self.register_buffer("zero", torch.zeros(1))
        self.register_buffer("one", torch.ones(1))

    def forward(self, image, background):  # pylint: disable=empty-docstring
        """"""
        z = self.enc.forward(image - background)
        recon_mean = self.dec.forward(z)
        return recon_mean + background

    def get_loss(self, image, recon_mean):
        return -Normal(recon_mean, recon_mean.sqrt()).log_prob(image).sum()

    # ---------------
    # Optimizer
    # ----------------

    def configure_optimizers(self):  # pylint: disable=empty-docstring
        """"""
        assert self.hparams["optimizer_params"] is not None, "Need to specify 'optimizer_params'."
        name = self.hparams["optimizer_params"]["name"]
        kwargs = self.hparams["optimizer_params"]["kwargs"]
        return get_optimizer(name, self.parameters(), kwargs)

    # ---------------
    # Training
    # ----------------

    def training_step(self, batch, batch_idx):  # pylint: disable=unused-argument,empty-docstring
        """"""
        images, background = batch["images"], batch["background"]
        recon_mean = self(images, background)
        loss = self.get_loss(images, recon_mean)
        self.log("train_loss", loss)
        return loss

    # ---------------
    # Validation
    # ----------------

    def validation_step(self, batch, batch_idx):  # pylint: disable=unused-argument,empty-docstring
        """"""
        images, background = batch["images"], batch["background"]
        recon_mean = self(images, background)
        loss = self.get_loss(images, recon_mean)

        # metrics
        self.log("val_loss", loss)
        residuals = (images - recon_mean) / torch.sqrt(images)
        self.log("val_max_residual", residuals.abs().max())
        return {"images": images, "recon_mean": recon_mean}

    def validation_epoch_end(self, outputs):  # pylint: disable=empty-docstring
        """"""
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

    def test_step(self, batch, batch_idx):  # pylint: disable=unused-argument,empty-docstring
        """"""
        images, background = batch["images"], batch["background"]
        recon_mean = self(images, background)
        residuals = (images - recon_mean) / torch.sqrt(images)
        self.log("max_residual", residuals.abs().max())
