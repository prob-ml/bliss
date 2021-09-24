import pytorch_lightning as pl
import torch
from einops import rearrange
from matplotlib import pyplot as plt
from torch import nn
from torch.distributions import Normal
from torch.nn import functional as F
from torch.nn.modules.conv import Conv2d, ConvTranspose2d
from torch.nn.modules.module import Module

from bliss.optimizer import get_optimizer
from bliss.utils import make_grid

plt.switch_backend("Agg")


class CenteredGalaxyEncoder(nn.Module):
    def __init__(self, slen=53, latent_dim=8, n_bands=1, hidden=32):

        super().__init__()

        self.slen = slen
        self.latent_dim = latent_dim

        f = lambda x: (x - 5) // 3 + 1  # function to figure out dimension of conv2d output.
        min_slen = f(slen)

        # self.features = nn.Sequential(
        #     nn.Conv2d(n_bands, 4, 5, stride=3, padding=0),
        #     nn.LeakyReLU(),
        #     nn.Flatten(),
        #     nn.Linear(min_slen * min_slen * 4, hidden * 16),
        #     nn.LeakyReLU(),
        #     nn.Linear(hidden * 16, hidden * 4),
        #     nn.LeakyReLU(),
        #     nn.Linear(hidden * 4, hidden),
        #     nn.LeakyReLU(),
        #     nn.Linear(hidden, latent_dim),
        # )
        self.features = nn.Sequential(
            ResidualConvDownsampleBlock(n_bands, 8, 3, 4),
            nn.LeakyReLU(),
            ResidualConvDownsampleBlock(n_bands*2, 8, 3, 4),
            nn.LeakyReLU(),
            ResidualConvDownsampleBlock(n_bands*4, 8, 3, 4),
            nn.LeakyReLU(),
            ResidualConvDownsampleBlock(n_bands*8, 8, 3, 4),
        )

    def forward(self, image):
        """Encodes galaxy from image."""
        # print(image.shape)
        return self.features(image)


class CenteredGalaxyDecoder(nn.Module):
    def __init__(self, slen=53, latent_dim=8, n_bands=1, hidden=32):
        super().__init__()

        self.slen = slen

        f = lambda x: (x - 5) // 3 + 1  # function to figure out dimension of conv2d output.
        g = lambda x: (x - 1) * 3 + 5
        self.min_slen = f(slen)
        assert g(self.min_slen) == slen

        # self.fc = nn.Sequential(
        #     nn.Linear(latent_dim, hidden),
        #     nn.LeakyReLU(),
        #     nn.Linear(hidden, hidden * 4),
        #     nn.LeakyReLU(),
        #     nn.Linear(hidden * 4, hidden * 16),
        #     nn.LeakyReLU(),
        #     nn.Linear(hidden * 16, self.min_slen * self.min_slen * 4),
        # )

        # self.features = nn.Sequential(nn.ConvTranspose2d(4, n_bands, 5, stride=3))
        output_padding = [0, 1, 1, 0]
        self.features = nn.Sequential(
            ResidualConvUpsampleBlock(n_bands*16, 8, 3, 4, output_padding[0]),
            nn.LeakyReLU(),
            ResidualConvUpsampleBlock(n_bands*8, 8, 3, 4, output_padding[1]),
            nn.LeakyReLU(),
            ResidualConvUpsampleBlock(n_bands*4, 8, 3, 4, output_padding[2]),
            nn.LeakyReLU(),
            ResidualConvUpsampleBlock(n_bands*2, 8, 3, 4, output_padding[3]),
        )

    def forward(self, z):
        """Decodes image from latent representation."""
        # z = self.fc(z)
        # z = rearrange(z, "b (c h w) -> b c h w", h=self.min_slen, w=self.min_slen)
        return self.features(z)


class OneCenteredGalaxyAE(pl.LightningModule):

    # ---------------
    # Model
    # ----------------

    def __init__(
        self,
        slen=53,
        latent_dim=8,
        hidden=32,
        n_bands=1,
        mse_residual_model_loss: bool = False,
        optimizer_params: dict = None,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.main_autoencoder = nn.Sequential(
            CenteredGalaxyEncoder(slen=slen, latent_dim=latent_dim, hidden=hidden, n_bands=n_bands),
            CenteredGalaxyDecoder(slen=slen, latent_dim=latent_dim, hidden=hidden, n_bands=n_bands),
        )
        self.residual_autoencoder = nn.Sequential(
            CenteredGalaxyEncoder(slen=slen, latent_dim=latent_dim, hidden=hidden, n_bands=n_bands),
            CenteredGalaxyDecoder(slen=slen, latent_dim=latent_dim, hidden=hidden, n_bands=n_bands),
        )
        self.mse_residual_model_loss = mse_residual_model_loss

        self.register_buffer("zero", torch.zeros(1))
        self.register_buffer("one", torch.ones(1))

    def _main_forward(self, image, background):
        return F.relu(self.main_autoencoder(image - background)) + background

    def _residual_forward(self, residual):
        return self.residual_autoencoder(residual)

    def forward(self, image, background):
        """Gets reconstructed image from running through encoder and decoder."""
        recon_mean_main = self._main_forward(image, background)
        recon_mean_residual = self._residual_forward(image - recon_mean_main)
        return recon_mean_main + recon_mean_residual

    def get_likelihood_loss(self, image, recon_mean):
        # this is nan whenever recon_mean is not strictly positive
        return -Normal(recon_mean, recon_mean.sqrt()).log_prob(image).sum()

    def get_residual_model_loss(self, image, recon_mean_main, recon_mean_residual):
        if self.mse_residual_model_loss:
            loss = F.mse_loss(image - recon_mean_main, recon_mean_residual)
        else:
            loss = self.get_likelihood_loss(image, recon_mean_main + recon_mean_residual)
        return loss

    # ---------------
    # Optimizer
    # ----------------

    def configure_optimizers(self):
        """Configures optimizers for training (pytorch lightning)."""
        assert self.hparams["optimizer_params"] is not None, "Need to specify `optimizer_params`."
        name = self.hparams["optimizer_params"]["name"]
        kwargs = self.hparams["optimizer_params"]["kwargs"]
        opt_main = get_optimizer(name, self.main_autoencoder.parameters(), kwargs)
        opt_residual = get_optimizer(name, self.residual_autoencoder.parameters(), kwargs)
        return opt_main, opt_residual

    def optimizer_step(
        self,
        epoch,
        batch_idx,
        optimizer,
        optimizer_idx,
        optimizer_closure,
        on_tpu=False,
        using_native_amp=False,
        using_lbfgs=False,
    ):
        # update generator every step
        if optimizer_idx == 0:
            optimizer.step(closure=optimizer_closure)

        if optimizer_idx == 1:
            if self.trainer.global_step > 500:
                optimizer.step(closure=optimizer_closure)
                # # the closure (which includes the `training_step`) will be executed by `optimizer.step`
                # optimizer.step(closure=optimizer_closure)

    # ---------------
    # Training
    # ----------------

    def training_step(self, batch, batch_idx, optimizer_idx):
        """Training step (pytorch lightning)."""
        images, background = batch["images"], batch["background"]
        if optimizer_idx == 0:
            recon_mean_main = self._main_forward(images, background)
            # with torch.no_grad():
            #     recon_mean_residual = self._residual_forward(images - recon_mean_main)
            # loss = self.get_likelihood_loss(images - recon_mean_residual, recon_mean_main)
            loss = self.get_likelihood_loss(images, recon_mean_main)
            self.log("train/loss_main", loss, prog_bar=True)
        if optimizer_idx == 1:
            with torch.no_grad():
                recon_mean_main = self._main_forward(images, background)
            recon_mean_residual = self._residual_forward(images - recon_mean_main)
            loss = self.get_residual_model_loss(images, recon_mean_main, recon_mean_residual)
            self.log("train/loss_residual", loss, prog_bar=True)

            recon_mean_final = recon_mean_main + recon_mean_residual
            loss_final = self.get_likelihood_loss(images, recon_mean_final)
            self.log("train/loss", loss_final, prog_bar=True)
        return loss

    # ---------------
    # Validation
    # ----------------

    def validation_step(self, batch, batch_idx):
        """Validation step (pytorch lightning)."""
        images, background = batch["images"], batch["background"]
        recon_mean_main = self._main_forward(images, background)
        loss_main = self.get_likelihood_loss(images, recon_mean_main)
        self.log("val/loss_main", loss_main)

        recon_mean_residual = self._residual_forward(images - recon_mean_main)
        loss_residual = self.get_residual_model_loss(images, recon_mean_main, recon_mean_residual)
        self.log("val/loss_residual", loss_residual)

        recon_mean_final = recon_mean_main + recon_mean_residual
        loss = self.get_likelihood_loss(images, recon_mean_final)
        self.log("val/loss", loss)

        # metrics
        residuals = (images - recon_mean_final) / torch.sqrt(images)
        self.log("val/max_residual", residuals.abs().max())
        return {
            "images": images,
            "recon_mean_main": recon_mean_main,
            "recon_mean_residual": recon_mean_residual,
            "recon_mean_final": recon_mean_final,
        }

    def validation_epoch_end(self, outputs):
        """Validation epoch end (pytorch lightning)."""
        if self.logger:
            images = torch.cat([x["images"] for x in outputs])
            recon_mean_final = torch.cat([x["recon_mean_final"] for x in outputs])
            recon_mean_main = torch.cat([x["recon_mean_main"] for x in outputs])
            recon_mean_residual = torch.cat([x["recon_mean_residual"] for x in outputs])

            reconstructions = self.plot_reconstruction(
                images, recon_mean_main, recon_mean_residual, recon_mean_final
            )
            grid_example = self.plot_grid_examples(images, recon_mean_final)

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

    def plot_reconstruction(self, images, recon_mean_main, recon_mean_residual, recon_mean_final):

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

        residuals_main = (images - recon_mean_main) / torch.sqrt(images)
        residuals_final = (images - recon_mean_final) / torch.sqrt(images)
        recon_mean_residual = recon_mean_residual / torch.sqrt(images)

        residuals_idx = residuals_final.abs().mean(dim=(1, 2, 3)).argsort(descending=True)
        large_residuals_idx = residuals_idx[: num_examples // 2]
        small_residuals_idx = residuals_idx[-num_examples // 2 :]
        plot_idx = torch.cat((large_residuals_idx, small_residuals_idx))

        images = images[plot_idx]
        recon_mean_main = recon_mean_main[plot_idx]
        recon_mean_residual = recon_mean_residual[plot_idx]
        recon_mean_final = recon_mean_final[plot_idx]
        residuals_main = residuals_main[plot_idx]
        residuals_final = residuals_final[plot_idx]

        residual_vmax = torch.ceil(residuals_final.max().cpu()).numpy()
        residual_vmin = torch.floor(residuals_final.min().cpu()).numpy()

        plt.ioff()

        fig = plt.figure(figsize=(15, 25))
        for i in range(num_examples):
            image = images[i, 0].data.cpu()
            recon_final = recon_mean_final[i, 0].data.cpu()
            recon_main = recon_mean_main[i, 0].data.cpu()
            recon_residual = recon_mean_residual[i, 0].data.cpu()
            res_main = residuals_main[i, 0].data.cpu()
            res_final = residuals_final[i, 0].data.cpu()

            image_vmax = torch.ceil(torch.max(image.max(), recon_final.max())).cpu().numpy()
            image_vmin = torch.floor(torch.min(image.min(), recon_final.min())).cpu().numpy()

            plots = {
                "images": image,
                "recon_mean_main": recon_main,
                "residuals_main": res_main,
                "recon_mean_residual": recon_residual,
                "recon_mean_final": recon_final,
                "residuals_final": res_final,
            }

            for j, (title, plot) in enumerate(plots.items()):
                plt.subplot(num_examples, num_cols, num_cols * i + j + 1)
                plt.title(title)
                if "residual" in title:
                    vmin, vmax = residual_vmin, residual_vmax
                else:
                    vmin, vmax = image_vmin, image_vmax
                plt.imshow(plot.numpy(), interpolation=None, vmin=vmin, vmax=vmax)
                plt.colorbar()
        plt.tight_layout()

        return fig

    def test_step(self, batch, batch_idx):
        """Testing step (pytorch lightning)."""
        images, background = batch["images"], batch["background"]
        recon_mean = self(images, background)
        residuals = (images - recon_mean) / torch.sqrt(images)
        self.log("max_residual", residuals.abs().max())

class ResConv2dBlock(Conv2d):
    def forward(self, input):
        y = super().forward(input)
        y = F.relu(y)
        return input + y

class ResidualConvDownsampleBlock(nn.Module):
    def __init__(self, in_channels, expand_factor, kernel_size, n_layers):
        super().__init__()
        expand_channels = in_channels*expand_factor
        out_channels = in_channels*2
        conv_initial = Conv2d(in_channels, expand_channels, kernel_size, stride=1, padding=1)
        conv = Conv2d(expand_channels, expand_channels, kernel_size, stride=2)
        layers = [conv_initial, nn.ReLU(), conv, nn.ReLU()]
        for _ in range(n_layers - 1):
            layers.append(ResConv2dBlock(expand_channels, expand_channels, kernel_size, stride=1, padding=1))
        layers.append(Conv2d(expand_channels, out_channels, kernel_size, stride=1, padding=1))
        self.f = nn.Sequential(*layers)
        # self.downsample = nn.Pool2d(kernel_size, stride=2)
    def forward(self, x):
        y = self.f(x)
        x_downsampled = F.interpolate(x, size=y.shape[-2:], mode="bilinear", align_corners=True)
        # x_downsampled = self.downsample(x)
        # x_downsampled = x_downsampled * (x.max() / x_downsampled.max())
        # print(x.shape)
        # print(x.mean())
        # print(x_downsampled.mean())
        x_downsampled = x_downsampled.repeat(1, 2, 1, 1)
        # print(y.shape)
        return y + x_downsampled
        # return x_downsampled


class ResidualConvUpsampleBlock(nn.Module):
    def __init__(self, in_channels, expand_factor, kernel_size, n_layers, output_padding):
        super().__init__()
        expand_channels = in_channels*expand_factor
        out_channels = in_channels//2
        conv_initial = Conv2d(in_channels, expand_channels, kernel_size, stride=1, padding=1)
        conv = ConvTranspose2d(expand_channels, expand_channels, kernel_size, stride=2, output_padding=output_padding)
        layers = [conv_initial, conv]
        for _ in range(n_layers - 1):
            # layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU())
            layers.append(ResConv2dBlock(expand_channels, expand_channels, kernel_size, stride=1, padding=1))
        layers.append(nn.ReLU())
        layers.append(Conv2d(expand_channels, out_channels, kernel_size, stride=1, padding=1))
        self.f = nn.Sequential(*layers)
    def forward(self, x):
        y = self.f(x)
        x_upsampled = F.interpolate(x, size=y.shape[-2:], mode="nearest")
        x_upsampled = x_upsampled[:, :y.shape[1], :, :]
        # print(y.shape)
        return y + x_upsampled
        # return x_upsampled


# class ResidualConvCell(nn.Module):
#     def __init__(self, in_channels, mode, n_layers=2):
#         super().__init__()
#         ## First layer is either a downsample or upsample
#         if mode == 'down':
#             out_channels = in_channels*2
#             conv1 = Conv2d(in_channels, out_channels, kernel_size=(3,3), stride=2)
#         elif mode == 'up':
#             out_channels = in_channels//2
#             conv1 = ConvTranspose2d(in_channels, out_channels, kernel_size=(3,3), stride=2)
#         elif mode == 'same':
#             conv1 = Conv2d(in_channels, in_channels, kernel_size=(3,3), stride=1)
#         else:
#             raise ValueError("mode needs to be one of `down`, `up`, or `same`.")
        
#         layers = [conv1]
#         # layers.append(nn.BatchNorm2d())
#         # layers.append(nn.ReLU())

#         for n in range(n_layers - 1):
#             layers.append(nn.BatchNorm2d(out_channels))
#             layers.append(nn.ReLU())
#             layers.append(Conv2d(in_channels, out_channels))



# class ResidualConvBlock(nn.Module):
#     def __init__(self, *args, **kwargs):
#         super().__init__()
#         conv = Conv2d(*args, **kwargs)
#         layers = [conv]
#         layers.append(nn.BatchNorm2d())
#         layers.append(nn.ReLU())
#         self.f = nn.Sequential(*layers)
#     def forward(self, x):
#         y = self.f(x) + x
#         return y
