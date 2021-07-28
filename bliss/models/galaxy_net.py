import pytorch_lightning as pl
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

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

    def forward(self, image):
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

    def __init__(
        self,
        slen=53,
        latent_dim=8,
        hidden=256,
        n_bands=1,
        optimizer_params: dict = None,
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

        # FIXME it's available through self.hparams["optimizer_params"]
        self.optimizer_params = optimizer_params

    def forward(self, image, background):
        z = self.enc.forward(image - background)
        recon_mean = self.dec.forward(z)
        recon_mean = recon_mean + background
        return recon_mean

    def get_loss(self, image, recon_mean):
        return -Normal(recon_mean, recon_mean.sqrt()).log_prob(image).sum()

    # ---------------
    # Optimizer
    # ----------------

    def configure_optimizers(self):
        assert self.optimizer_params is not None, "Need to specify `optimizer_params`."
        name = self.optimizer_params["name"]
        kwargs = self.optimizer_params["kwargs"]
        return get_optimizer(name, self.parameters(), kwargs)

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
        # FIXME: change to val/loss so it's automatically grouped in tensorboard
        self.log("val_loss", loss)
        residuals = (images - recon_mean) / torch.sqrt(images)
        self.log("val_max_residual", residuals.abs().max())
        return {"images": images, "recon_mean": recon_mean}

    def validation_epoch_end(self, outputs):
        images = torch.cat([x["images"] for x in outputs])
        recon_mean = torch.cat([x["recon_mean"] for x in outputs])
        fig = self.plot_reconstruction(images, recon_mean)
        if self.logger:
            self.logger.experiment.add_figure(f"Images {self.current_epoch}", fig)

    def plot_reconstruction(self, images, recon_mean):

        # 1. only plot i band if available, otherwise the highest band given.
        # 2. plot `num_examples//2` images with the largest average residual
        #    and `num_examples//2` images with the smallest average residual
        #    across all batches in the last epoch
        # 3. residual color range (`vmin`, `vmax`) are fixed across all samples
        #    (same across all rows in the subplot grid)
        # 4. image and recon_mean color range are fixed for their particular sample
        #    (same within each row in the subplot grid)

        assert images.size(0) >= 10
        num_examples = 10
        num_cols = 3

        residuals = (images - recon_mean) / torch.sqrt(images)
        residuals_idx = residuals.abs().mean(dim=(1, 2, 3)).argsort(descending=True)
        large_residuals_idx = residuals_idx[: num_examples // 2]
        small_residuals_idx = residuals_idx[-num_examples // 2 :]
        plot_idx = torch.cat((large_residuals_idx, small_residuals_idx))

        images = images[plot_idx]
        recon_mean = recon_mean[plot_idx]
        residuals = residuals[plot_idx]

        residual_vmax = torch.ceil(residuals.max().cpu()).numpy()
        residual_vmin = torch.floor(residuals.min().cpu()).numpy()
        plt.ioff()

        fig = plt.figure(figsize=(10, 25))

        for i in range(num_examples):
            image = images[i, 0].data.cpu()
            recon = recon_mean[i, 0].data.cpu()

            vmax = torch.ceil(torch.max(image.max(), recon.max())).cpu().numpy()
            vmin = torch.floor(torch.min(image.min(), recon.min())).cpu().numpy()

            plt.subplot(num_examples, num_cols, num_cols * i + 1)
            plt.title("images")
            plt.imshow(image.numpy(), interpolation=None, vmin=vmin, vmax=vmax)
            plt.colorbar()

            plt.subplot(num_examples, num_cols, num_cols * i + 2)
            plt.title("recon_mean")
            plt.imshow(recon.numpy(), interpolation=None, vmin=vmin, vmax=vmax)
            plt.colorbar()

            plt.subplot(num_examples, num_cols, num_cols * i + 3)
            res = residuals[i, 0].data.cpu().numpy()
            if i < num_examples // 2:
                plt.title(f"residuals (worst), avg abs residual: {abs(res).mean():.4f}")
            else:
                plt.title(f"residuals (best), avg abs residual: {abs(res).mean():.4f}")
            plt.imshow(
                res,
                interpolation=None,
                vmin=residual_vmin,
                vmax=residual_vmax,
            )
            plt.colorbar()

        plt.tight_layout()

        return fig

    def test_step(self, batch, batch_idx):  # pylint: disable=unused-argument
        images, background = batch["images"], batch["background"]
        recon_mean = self(images, background)
        residuals = (images - recon_mean) / torch.sqrt(images)
        self.log("max_residual", residuals.abs().max())
