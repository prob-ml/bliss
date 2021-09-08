import pytorch_lightning as pl
import matplotlib.pyplot as plt

import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Normal

from bliss.optimizer import get_optimizer
from bliss.utils import make_grid

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
        # self.init_weight()

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="leaky_relu")

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
        # self.init_weight()

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="leaky_relu")

    def forward(self, z):
        z = self.fc(z)
        z = z.view(-1, 8, self.min_slen, self.min_slen)
        z = self.deconv(z)
        z = z[:, :, : self.slen, : self.slen]
        assert z.shape[-1] == self.slen and z.shape[-2] == self.slen
        recon_mean = z
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
        optimizer_params: dict = None,  # pylint: disable=unused-argument
    ):
        super().__init__()
        self.save_hyperparameters()

        self.enc_0 = CenteredGalaxyEncoder(
            slen=slen, latent_dim=latent_dim, hidden=hidden, n_bands=n_bands
        )
        self.dec_0 = CenteredGalaxyDecoder(
            slen=slen, latent_dim=latent_dim, hidden=hidden, n_bands=n_bands
        )

        self.enc_1 = CenteredGalaxyEncoder(
            slen=slen, latent_dim=latent_dim, hidden=hidden, n_bands=n_bands
        )
        self.dec_1 = CenteredGalaxyDecoder(
            slen=slen, latent_dim=latent_dim, hidden=hidden, n_bands=n_bands
        )

        self.register_buffer("zero", torch.zeros(1))
        self.register_buffer("one", torch.ones(1))

    def forward_model_0(self, image, background):
        z = self.enc_0.forward(image - background)
        recon_mean = F.relu(self.dec_0.forward(z))
        recon_mean = recon_mean + background
        return recon_mean

    def forward_model_1(self, residual):
        z = self.enc_1.forward(residual)
        recon_mean = F.leaky_relu(self.dec_1.forward(z))
        return recon_mean

    def forward(self, image, background):
        recon_mean_0 = self.forward_model_0(image, background)
        recon_mean_1 = self.forward_model_1(image - recon_mean_0)
        recon_mean = recon_mean_0 + recon_mean_1

        return recon_mean

    def get_loss(self, image, recon_mean):
        # this is nan whenever recon_mean is not strictly positive
        return -Normal(recon_mean, recon_mean.sqrt()).log_prob(image).sum()

    # ---------------
    # Optimizer
    # ----------------

    def configure_optimizers(self):
        assert self.hparams["optimizer_params"] is not None, "Need to specify `optimizer_params`."
        name = self.hparams["optimizer_params"]["name"]
        kwargs = self.hparams["optimizer_params"]["kwargs"]
        opt_0_params = list(self.enc_0.parameters()) + list(self.dec_0.parameters())
        opt_1_params = list(self.enc_1.parameters()) + list(self.dec_1.parameters())
        opt_0 = get_optimizer(name, opt_0_params, kwargs)
        opt_1 = get_optimizer(name, opt_1_params, kwargs)
        return opt_0, opt_1

    # ---------------
    # Training
    # ----------------

    def training_step(self, batch, batch_idx, optimizer_idx):  # pylint: disable=unused-argument
        images, background = batch["images"], batch["background"]
        if optimizer_idx == 0:
            recon_mean_0 = self.forward_model_0(images, background)
            loss_0 = self.get_loss(images, recon_mean_0)
            self.log("train/loss_0", loss_0, prog_bar=True)
            return loss_0
        if optimizer_idx == 1:
            with torch.no_grad():
                recon_mean_0 = self.forward_model_0(images, background)
            recon_mean_1 = self.forward_model_1(images - recon_mean_0)
            loss_1 = F.mse_loss(images - recon_mean_0, recon_mean_1)
            self.log("train/loss_1", loss_1, prog_bar=True)

            recon_mean = recon_mean_0 + recon_mean_1
            loss = self.get_loss(images, recon_mean)
            self.log("train/loss", loss, prog_bar=True)
            return loss_1

    # ---------------
    # Validation
    # ----------------

    def validation_step(self, batch, batch_idx):  # pylint: disable=unused-argument
        images, background = batch["images"], batch["background"]
        recon_mean_0 = self.forward_model_0(images, background)
        loss_0 = self.get_loss(images, recon_mean_0)
        self.log("val/loss_0", loss_0)

        recon_mean_1 = self.forward_model_1(images - recon_mean_0)
        loss_1 = F.mse_loss(images - recon_mean_0, recon_mean_1)
        self.log("val/loss_1", loss_1)

        recon_mean = recon_mean_0 + recon_mean_1
        loss = self.get_loss(images, recon_mean)
        self.log("val/loss", loss)

        # metrics
        residuals = (images - recon_mean) / torch.sqrt(images)
        self.log("val/max_residual", residuals.abs().max())
        return {
            "images": images,
            "recon_mean_0": recon_mean_0,
            "recon_mean_1": recon_mean_1,
            "recon_mean": recon_mean,
        }

    def validation_epoch_end(self, outputs):
        if self.logger:
            images = torch.cat([x["images"] for x in outputs])
            recon_mean = torch.cat([x["recon_mean"] for x in outputs])
            recon_mean_0 = torch.cat([x["recon_mean_0"] for x in outputs])
            recon_mean_1 = torch.cat([x["recon_mean_1"] for x in outputs])

            reconstructions = self.plot_reconstruction(
                images, recon_mean_0, recon_mean_1, recon_mean
            )
            grid_example = self.plot_grid_examples(images, recon_mean)

            self.logger.experiment.add_figure(f"Epoch:{self.current_epoch}/images", reconstructions)
            self.logger.experiment.add_figure(
                f"Epoch:{self.current_epoch}/grid_examples", grid_example
            )

    def plot_grid_examples(self, images, recon_mean):
        # 1.  plot a grid of all input images and recon_mean
        # 2.  only plot the highest band

        nrow = 16
        residuals = (images - recon_mean) / torch.sqrt(images)

        image_grid = make_grid(images, nrow=nrow)[0]
        recon_grid = make_grid(recon_mean, nrow=nrow)[0]
        residual_grid = make_grid(residuals, nrow=nrow)[0]
        h, w = image_grid.size()
        base_size = 8
        fig = plt.figure(figsize=(3 * base_size, int(h / w * base_size)))
        for i, grid in enumerate([image_grid, recon_grid, residual_grid]):
            plt.subplot(1, 3, i + 1)
            plt.imshow(grid.cpu().numpy(), interpolation=None)
            if i == 0:
                plt.title("images")
            elif i == 1:
                plt.title("recon_mean")
            else:
                plt.title("residuals")
            plt.xticks([])
            plt.yticks([])
        return fig

    def plot_reconstruction(self, images, recon_mean_0, recon_mean_1, recon_mean):

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
        num_cols = 6

        residuals_0 = (images - recon_mean_0) / torch.sqrt(images)
        residuals = (images - recon_mean) / torch.sqrt(images)

        residuals_idx = residuals.abs().mean(dim=(1, 2, 3)).argsort(descending=True)
        large_residuals_idx = residuals_idx[: num_examples // 2]
        small_residuals_idx = residuals_idx[-num_examples // 2 :]
        plot_idx = torch.cat((large_residuals_idx, small_residuals_idx))

        images = images[plot_idx]
        recon_mean_0 = recon_mean_0[plot_idx]
        recon_mean_1 = recon_mean_1[plot_idx]
        recon_mean = recon_mean[plot_idx]
        residuals = residuals[plot_idx]
        residuals_0 = residuals_0[plot_idx]

        residual_vmax = torch.ceil(residuals.max().cpu()).numpy()
        residual_vmin = torch.floor(residuals.min().cpu()).numpy()

        plt.ioff()

        fig = plt.figure(figsize=(15, 25))

        for i in range(num_examples):
            image = images[i, 0].data.cpu()
            recon = recon_mean[i, 0].data.cpu()
            recon_0 = recon_mean_0[i, 0].data.cpu()
            recon_1 = recon_mean_1[i, 0].data.cpu()

            vmax = torch.ceil(torch.max(image.max(), recon.max())).cpu().numpy()
            vmin = torch.floor(torch.min(image.min(), recon.min())).cpu().numpy()

            plt.subplot(num_examples, num_cols, num_cols * i + 1)
            plt.title("images")
            plt.imshow(image.numpy(), interpolation=None, vmin=vmin, vmax=vmax)
            plt.colorbar()

            plt.subplot(num_examples, num_cols, num_cols * i + 2)
            plt.title("recon_mean_0")
            plt.imshow(recon_0.numpy(), interpolation=None, vmin=vmin, vmax=vmax)
            plt.colorbar()

            plt.subplot(num_examples, num_cols, num_cols * i + 3)
            res_0 = residuals_0[i, 0].data.cpu().numpy()
            if i < num_examples // 2:
                plt.title(f"worst residuals_0, avg abs residual: {abs(res_0).mean():.3f}")
            else:
                plt.title(f"best residuals_0, avg abs residual: {abs(res_0).mean():.3f}")
            plt.imshow(
                res_0,
                interpolation=None,
                vmin=residual_vmin,
                vmax=residual_vmax,
            )

            plt.colorbar()
            plt.subplot(num_examples, num_cols, num_cols * i + 4)
            plt.title("recon_mean_1")
            plt.imshow(recon_1.numpy(), interpolation=None)
            plt.colorbar()

            plt.subplot(num_examples, num_cols, num_cols * i + 5)
            plt.title("recon_mean")
            plt.imshow(recon.numpy(), interpolation=None, vmin=vmin, vmax=vmax)
            plt.colorbar()

            plt.subplot(num_examples, num_cols, num_cols * i + 6)
            res = residuals[i, 0].data.cpu().numpy()
            if i < num_examples // 2:
                plt.title(f"worst residuals, avg abs residual: {abs(res).mean():.3f}")
            else:
                plt.title(f"best residuals, avg abs residual: {abs(res).mean():.3f}")
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
