import math

import pytorch_lightning as pl
import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.distributions import Normal
from torch.nn import functional as F
from torch.nn.modules.batchnorm import BatchNorm1d
from torch.nn.modules.conv import Conv2d, ConvTranspose2d
from torch.nn.utils import weight_norm as wn
from tqdm import tqdm

from bliss.optimizer import load_optimizer
from bliss.reporting import plot_image
from bliss.utils import make_grid

plt.switch_backend("Agg")
plt.ioff()


class OneCenteredGalaxyAE(pl.LightningModule):
    """Autoencoder for single, centered galaxy images.

    This module implements an autoencoder(AE) + training procedure on images of centered
    galaxies. The architecture consists of a "main" AE followed by a "residual" AE.
    The main AE is trained to minimize loss on its own.
    The residual AE, whose reconstructed is added to the main autoencoder,
    is trained to minimize the entire loss.

    Attributes:
        main_autoencoder: The first, "main" AE.
        main_encoder: The encoder from the first, "main" AE.
        main_decoder: The decoder from the first, "main" AE.
        residual_autoencoder: The second, "residual" AE.
        residual_encoder: The encoder from the second, "residual" AE.
        residual_decoder: The decoder from the second, "residual" AE.
        slen: Side length of input images.
        latent_dim: Latent dimension of encoded representation.
        min_sd: Minimum sd for log-likelihood.
        residual_delay_n_steps: Number of training steps before starting to
            train residual model.
    """

    def __init__(
        self,
        slen: int = 53,
        latent_dim: int = 16,
        n_bands: int = 1,
        residual_delay_n_steps: int = 500,
        optimizer_params: dict = None,
        min_sd=1e-3,
    ):
        """Initializer.

        Args:
            slen: (optional) Image side length.
            latent_dim: (optional) Latent size of each autoencoder.
            n_bands: (optional) Number of bands in image.
            residual_delay_n_steps: (optional)
                Number of training steps before starting to train residual model.
            optimizer_params: (optional)
                Parameters used to construct training optimizer.
            min_sd: (optional)
                Minimum sd for log-likelihood.
        """
        super().__init__()
        self.save_hyperparameters()

        self.main_encoder = CenteredGalaxyEncoder(
            slen=slen,
            latent_dim=(latent_dim // 2) * 2,
            n_bands=n_bands,
        )
        self.main_decoder = CenteredGalaxyDecoder(
            slen=slen, latent_dim=latent_dim // 2, n_bands=n_bands
        )
        self.main_autoencoder = nn.Sequential(self.main_encoder, self.main_decoder)

        self.residual_encoder = CenteredGalaxyEncoder(
            slen=slen,
            latent_dim=(latent_dim // 2) * 2,
            n_bands=n_bands,
        )
        self.residual_decoder = CenteredGalaxyDecoder(
            slen=slen, latent_dim=latent_dim // 2, n_bands=n_bands
        )
        self.residual_autoencoder = nn.Sequential(self.residual_encoder, self.residual_decoder)

        self.dist_main = Normal(0.0, 1.0)
        self.dist_residual = Normal(0.0, 1.0)

        self.residual_delay_n_steps = residual_delay_n_steps
        assert slen == 53, "Currently slen is fixed at 53"
        self.slen = slen
        self.latent_dim = latent_dim
        self.min_sd = min_sd

    def forward(self, image, background):
        """Gets reconstructed image from running through encoder and decoder."""
        recon_mean_main, pq_latent_main = self._main_forward(image, background)
        recon_mean_residual, pq_latent_residual = self._residual_forward(image - recon_mean_main)
        return recon_mean_main + recon_mean_residual, pq_latent_main + pq_latent_residual

    def get_encoder(self, allow_pad=False):
        return OneCenteredGalaxyEncoder(
            self.main_encoder,
            self.main_decoder,
            self.residual_encoder,
            slen=self.slen,
            allow_pad=allow_pad,
        )

    def get_decoder(self):
        return OneCenteredGalaxyDecoder(self.main_decoder, self.residual_decoder)

    def sample_latent(self, n_samples):
        latent_main = self.dist_main.sample(torch.Size((n_samples, self.latent_dim // 2)))
        latent_residual = self.dist_residual.sample(torch.Size((n_samples, self.latent_dim // 2)))
        z = torch.cat((latent_main, latent_residual), dim=-1)
        return z

    def sample(self, n_samples):
        z = self.sample_latent(n_samples)
        decoder = self.get_decoder()
        return decoder(z)

    def generate_latents(self, dataloader, n_batches):
        """Induces a latent distribution for a non-probabilistic autoencoder."""
        latent_list = []
        enc = self.get_encoder()
        with torch.no_grad():
            for _ in tqdm(range(n_batches)):
                galaxy = next(iter(dataloader))
                noiseless = galaxy["noiseless"].to(self.device)
                latent_batch = enc(noiseless, 0.0)
                latent_list.append(latent_batch)
        return torch.cat(latent_list, dim=0)

    def training_step(self, batch, batch_idx, optimizer_idx):
        """Training step (pytorch lightning)."""
        images, background = batch["images"], batch["background"]
        if optimizer_idx == 0:
            recon_mean_main, pq_latent_main = self._main_forward(images, background)
            loss_recon = self._get_likelihood_loss(images, recon_mean_main)
            loss_prior = -pq_latent_main.sum()
            loss = loss_recon + loss_prior
            self.log("train/loss_main", loss, prog_bar=True)
            self.log("train/loss_main_recon", loss_recon)
        if optimizer_idx == 1:
            with torch.no_grad():
                recon_mean_main, pq_latent_main = self._main_forward(images, background)
            recon_mean_residual, pq_latent_residual = self._residual_forward(
                images - recon_mean_main
            )
            recon_mean_final = F.relu(recon_mean_main + recon_mean_residual)
            loss_recon = self._get_likelihood_loss(images, recon_mean_final)
            loss_prior = -(pq_latent_main.sum() + pq_latent_residual.sum())
            loss = loss_recon + loss_prior
            self.log("train/loss", loss, prog_bar=True)
            self.log("train/loss_recon", loss_recon, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step (pytorch lightning)."""
        images, background = batch["images"], batch["background"]
        recon_mean_main, p_latent_main = self._main_forward(images, background)
        loss_recon_main = self._get_likelihood_loss(images, recon_mean_main)
        loss_prior_main = -p_latent_main.sum()
        loss_main = loss_recon_main + loss_prior_main
        self.log("val/loss_main", loss_main)

        recon_mean_residual, p_latent_residual = self._residual_forward(images - recon_mean_main)
        recon_mean_final = F.relu(recon_mean_main + recon_mean_residual)
        loss_recon = self._get_likelihood_loss(images, recon_mean_final)
        loss_prior = loss_prior_main - p_latent_residual.sum()
        loss = loss_recon + loss_prior
        self.log("val/loss", loss)

        # metrics
        residuals = (images - recon_mean_final) / torch.sqrt(recon_mean_final)
        residuals_main = (images - recon_mean_main) / torch.sqrt(recon_mean_main)
        recon_mean_residual = recon_mean_residual / torch.sqrt(recon_mean_main)
        self.log("val/max_residual", residuals.abs().max())

        # Aggregate posterior
        u_main = self.main_encoder(images - background)
        u_residual = self.residual_encoder(images - recon_mean_main)
        return {
            "images": images,
            "recon_mean_main": recon_mean_main,
            "recon_mean_residual": recon_mean_residual,
            "recon_mean": recon_mean_final,
            "residuals": residuals,
            "residuals_main": residuals_main,
            "u_main": u_main,
            "u_residual": u_residual,
        }

    def validation_epoch_end(self, outputs):
        """Validation epoch end (pytorch lightning)."""

        output_tensors = {
            label: torch.cat([output[label] for output in outputs]) for label in outputs[0]
        }

        fig_random = self._plot_reconstruction(output_tensors, mode="random")
        fig_worst = self._plot_reconstruction(output_tensors, mode="worst")
        grid_example = self._plot_grid_examples(output_tensors)

        base_size = 8
        agg_posterior = plt.figure(figsize=(base_size, base_size))
        u_main = output_tensors["u_main"].cpu().detach().numpy()
        plt.scatter(u_main[:, 0], u_main[:, 1])

        if self.logger:
            heading = f"Epoch:{self.current_epoch}"
            self.logger.experiment.add_figure(f"{heading}/Random Images", fig_random)
            self.logger.experiment.add_figure(f"{heading}/Worst Images", fig_worst)
            self.logger.experiment.add_figure(f"{heading}/grid_examples", grid_example)
            self.logger.experiment.add_figure(f"{heading}/Aggregate posterior", agg_posterior)

    def test_step(self, batch, batch_idx):
        """Testing step (pytorch lightning)."""
        images, background = batch["images"], batch["background"]
        recon_mean = self(images, background)
        residuals = (images - recon_mean) / torch.sqrt(images)
        self.log("max_residual", residuals.abs().max())

    def configure_optimizers(self):
        """Configures optimizers for training (pytorch lightning)."""
        opt_main = load_optimizer(self.main_autoencoder.parameters(), self.hparams)
        opt_residual = load_optimizer(self.residual_autoencoder.parameters(), self.hparams)
        return opt_main, opt_residual

    def optimizer_step(
        self,
        epoch: int = None,
        batch_idx: int = None,
        optimizer=None,
        optimizer_idx: int = None,
        optimizer_closure=None,
        on_tpu: bool = None,
        using_native_amp: bool = None,
        using_lbfgs: bool = None,
    ) -> None:

        # update generator every step
        if optimizer_idx == 0:
            optimizer.step(closure=optimizer_closure)

        if optimizer_idx == 1:
            if self.trainer.global_step > self.residual_delay_n_steps:
                optimizer.step(closure=optimizer_closure)

    def _main_forward(self, image, background):
        latent_main_mean, latent_main_sd = torch.split(
            self.main_encoder(image - background), (self.latent_dim // 2, self.latent_dim // 2), -1
        )
        latent_dist = Normal(latent_main_mean, F.softplus(latent_main_sd))
        latent_main = latent_dist.rsample()
        p_latent_main = self.dist_main.log_prob(latent_main)
        q_latent_main = latent_dist.log_prob(latent_main)
        recon_mean_main = F.relu(self.main_decoder(latent_main)) + background
        return recon_mean_main, p_latent_main - q_latent_main

    def _residual_forward(self, residual):
        latent_residual_mean, latent_residual_sd = torch.split(
            self.residual_encoder(residual), (self.latent_dim // 2, self.latent_dim // 2), -1
        )
        latent_dist = Normal(latent_residual_mean, F.softplus(latent_residual_sd))
        latent_residual = latent_dist.rsample()
        p_latent_residual = self.dist_residual.log_prob(latent_residual)
        q_latent_residual = latent_dist.log_prob(latent_residual)
        recon_mean_residual = self.residual_decoder(latent_residual)
        return recon_mean_residual, p_latent_residual - q_latent_residual

    def _get_likelihood_loss(self, image, recon_mean):
        # this is nan whenever recon_mean is not strictly positive
        return -Normal(recon_mean, recon_mean.sqrt().clamp(min=self.min_sd)).log_prob(image).sum()

    def _plot_reconstruction(self, outputs, n_examples=10, mode="random", width=20, pad=6.0):
        # combine all images and recon_mean's into a single tensor
        images = outputs["images"]
        recon_mean = outputs["recon_mean"]
        residuals = outputs["residuals"]
        recon_mean_main = outputs["recon_mean_main"]
        residuals_main = outputs["residuals_main"]
        recon_mean_residual = outputs["recon_mean_residual"]

        # only plot i band if available, otherwise the highest band given.
        assert images.size(0) >= n_examples
        assert images.shape[1] == recon_mean.shape[1] == residuals.shape[1] == 1, "1 band only."
        figsize = (width, width * n_examples / 6)
        fig, axes = plt.subplots(nrows=n_examples, ncols=6, figsize=figsize)

        if mode == "random":
            indices = torch.randint(0, len(images), size=(n_examples,))
        elif mode == "worst":
            # get indices where absolute residual is the largest.
            absolute_residual = residuals.abs().sum(axis=(1, 2, 3))
            indices = absolute_residual.argsort()[-n_examples:]
        else:
            raise NotImplementedError(f"Specified mode '{mode}' has not been implemented.")

        # pick standard ranges for residuals
        vmin_res = residuals[indices].min().item()
        vmax_res = residuals[indices].max().item()

        for i in range(n_examples):
            idx = indices[i]

            ax_true = axes[i, 0]
            ax_recon = axes[i, 1]
            ax_res = axes[i, 2]
            ax_recon_main = axes[i, 3]
            ax_res_main = axes[i, 4]
            ax_recon_res = axes[i, 5]

            # only add titles to the first axes.
            if i == 0:
                ax_true.set_title("Images $x$", pad=pad)
                ax_recon.set_title(r"Reconstruction $\tilde{x}$", pad=pad)
                ax_res.set_title(
                    r"Residual $\left(x - \tilde{x}\right) / \sqrt{\tilde{x}}$", pad=pad
                )
                ax_recon_main.set_title("Initial Reconstruction", pad=pad)
                ax_res_main.set_title("Initial Residual", pad=pad)
                ax_recon_res.set_title("Initial Residual Reconstruction", pad=pad)

            # standarize ranges of true and reconstruction
            image = images[idx, 0].detach().cpu().numpy()
            recon = recon_mean[idx, 0].detach().cpu().numpy()
            residual = residuals[idx, 0].detach().cpu().numpy()
            recon_mean_main_i = recon_mean_main[idx, 0].detach().cpu().numpy()
            residuals_main_i = residuals_main[idx, 0].detach().cpu().numpy()
            recon_mean_residual_i = recon_mean_residual[idx, 0].detach().cpu().numpy()
            vmin = min(image.min().item(), recon.min().item())
            vmax = max(image.max().item(), recon.max().item())

            # plot images
            plot_image(fig, ax_true, image, vrange=(vmin, vmax))
            plot_image(fig, ax_recon, recon, vrange=(vmin, vmax))
            plot_image(fig, ax_res, residual, vrange=(vmin_res, vmax_res))
            plot_image(fig, ax_recon_main, recon_mean_main_i, vrange=(vmin, vmax))
            plot_image(fig, ax_res_main, residuals_main_i, vrange=(vmin_res, vmax_res))
            plot_image(fig, ax_recon_res, recon_mean_residual_i, vrange=(vmin_res, vmax_res))

        plt.tight_layout()
        return fig

    def _plot_grid_examples(self, outputs):
        images, recon_mean = outputs["images"], outputs["recon_mean"]
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


class CenteredGalaxyEncoder(nn.Module):
    def __init__(self, slen=53, latent_dim=8, n_bands=1):

        super().__init__()

        self.slen = slen
        self.latent_dim = latent_dim

        kernels = [3, 3, 3, 3, 1]
        in_size = 2 ** len(kernels)
        layers = []
        for (i, kernel_size) in enumerate(kernels):
            layer = ResidualConvBlock(n_bands * (2 ** i), 8, kernel_size, 4, mode="downsample")
            layers.append(layer)
            if i < len(kernels) - 1:
                layers.append(nn.LeakyReLU())
        layers.append(nn.Flatten())
        layers.append(nn.BatchNorm1d(in_size))
        layers.append(ResidualDenseBlock(in_size, 3, latent_dim))
        layers.append(nn.BatchNorm1d(latent_dim))
        self.features = nn.Sequential(*layers)

    def forward(self, image):
        """Encodes galaxy from image."""
        return self.features(image)


class CenteredGalaxyDecoder(nn.Module):
    def __init__(self, slen=53, latent_dim=8, n_bands=1):
        super().__init__()

        self.slen = slen

        kernels = [3, 3, 3, 3, 1]
        in_size = 2 ** len(kernels)
        layers = []
        slen_current = slen
        for (i, kernel_size) in enumerate(kernels):
            output_padding = (slen_current - kernel_size) % 2 if (slen_current != 2) else 0
            layer = ResidualConvBlock(
                n_bands * (2 ** (i + 1)),
                8,
                kernel_size,
                4,
                mode="upsample",
                output_padding=output_padding,
            )
            layers.append(layer)
            if i < len(kernels) - 1:
                layers.append(nn.LeakyReLU())
            slen_current = math.floor((slen_current - kernel_size) / 2 + 1)
        layers.append(
            nn.Unflatten(
                -1, torch.Size((n_bands * (2 ** len(kernels)), slen_current, slen_current))
            )
        )
        layers.append(ResidualDenseBlock(latent_dim, 3, in_size))
        self.features = nn.Sequential(*layers[::-1])

    def forward(self, z):
        """Decodes image from latent representation."""
        return self.features(z)


class OneCenteredGalaxyEncoder(nn.Module):
    """Encoder part of OneCenteredGalaxyAE.

    This module isolates the part of OneCenteredGalaxyAE which encodes
    a latent representation of an image. Mostly used within larger modules
    such as the galaxy deblender.

    Notes:
        The `allow_pad` argument is used so that the same architecture
        can be used for encoding galaxies from Galsim and encoding galaxies in the deblender.
        It is often the case that the side length (`slen`) is smaller in the second setting
        than the first. Without padding, the downsampling architecture likely won't work.
    """

    def __init__(
        self,
        main_encoder: CenteredGalaxyEncoder,
        main_decoder: CenteredGalaxyDecoder,
        residual_encoder: CenteredGalaxyEncoder,
        slen: int = None,
        allow_pad: bool = False,
    ):
        """Initializer.

        Args:
            main_encoder: The main encoder
            main_decoder: The main decoder
            residual_encoder: The residual encoder
            slen: (optional) The side-length of the galaxy image. Only needed if allow_pad is True.
            allow_pad: (optional) Should padding be added to an image
                if its size is less than slen? Defaults to False.
        """
        super().__init__()
        self.main_encoder = main_encoder
        self.main_decoder = main_decoder
        self.residual_encoder = residual_encoder
        self.slen = slen
        self.allow_pad = allow_pad

    def forward(self, image, background=0, deterministic=False):
        assert image.shape[-2] == image.shape[-1]
        if self.allow_pad:
            if image.shape[-1] < self.slen:
                d = self.slen - image.shape[-1]
                lpad = d // 2
                upad = d - lpad
                min_val = image.min()
                image = F.pad(image, (lpad, upad, lpad, upad), value=min_val)
                if isinstance(background, torch.Tensor):
                    background = F.pad(background, (lpad, upad, lpad, upad))
        latent_main_params = self.main_encoder(image - background)
        d_main = latent_main_params.shape[-1] // 2
        latent_main_mean, latent_main_sd = torch.split(latent_main_params, (d_main, d_main), -1)
        if deterministic:
            latent_main = latent_main_mean
        else:
            latent_dist = Normal(latent_main_mean, F.softplus(latent_main_sd))
            latent_main = latent_dist.rsample()
        recon_mean_main = F.relu(self.main_decoder(latent_main)) + background
        latent_residual_params = self.residual_encoder(image - recon_mean_main)
        d_residual = latent_residual_params.shape[-1] // 2
        latent_residual_mean, latent_residual_sd = torch.split(
            latent_residual_params, (d_residual, d_residual), -1
        )
        if deterministic:
            latent_residual = latent_residual_mean
        else:
            latent_residual_dist = Normal(latent_residual_mean, F.softplus(latent_residual_sd))
            latent_residual = latent_residual_dist.rsample()
        return torch.cat((latent_main, latent_residual), dim=-1)


class OneCenteredGalaxyDecoder(nn.Module):
    """Decoder part of OneCenteredGalaxyAE."""

    def __init__(self, main_decoder: nn.Module, residual_decoder: nn.Module):
        """Initializer.

        Args:
            main_decoder: The main decoder.
            residual_decoder: The residual decoder.
        """
        super().__init__()
        self.main_decoder = main_decoder
        self.residual_decoder = residual_decoder

    def forward(self, latent):
        latent_main, latent_residual = torch.split(latent, latent.shape[-1] // 2, -1)
        recon_mean_main = F.relu(self.main_decoder(latent_main))
        recon_mean_residual = self.residual_decoder(latent_residual)
        return F.relu(recon_mean_main + recon_mean_residual)


class ResidualConvBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        expand_factor,
        kernel_size,
        n_layers,
        mode="downsample",
        output_padding=None,
    ):
        super().__init__()
        self.mode = mode
        expand_channels = in_channels * expand_factor
        padding = kernel_size // 2
        conv_initial = wn(
            Conv2d(in_channels, expand_channels, kernel_size, stride=1, padding=padding)
        )
        kernel_size_dim_change = max(kernel_size, 2)
        if self.mode == "downsample":
            conv = wn(Conv2d(expand_channels, expand_channels, kernel_size_dim_change, stride=2))
            out_channels = in_channels * 2
        elif self.mode == "upsample":
            assert output_padding is not None
            conv = wn(
                ConvTranspose2d(
                    expand_channels,
                    expand_channels,
                    kernel_size_dim_change,
                    stride=2,
                    output_padding=output_padding,
                )
            )
            out_channels = in_channels // 2
        layers = [conv_initial, nn.ReLU(), conv, nn.ReLU()]
        for _ in range(n_layers - 1):
            layers.append(
                wn(
                    ResConv2dBlock(
                        expand_channels, expand_channels, kernel_size, stride=1, padding=padding
                    )
                )
            )
        layers.append(
            wn(Conv2d(expand_channels, out_channels, kernel_size, stride=1, padding=padding))
        )
        self.f = nn.Sequential(*layers)

    def forward(self, x):
        y = self.f(x)
        if self.mode == "downsample":
            x_downsampled = F.interpolate(x, size=y.shape[-2:], mode="bilinear", align_corners=True)
            x_downsampled = x_downsampled.repeat(1, 2, 1, 1)
            x_trans = x_downsampled
        elif self.mode == "upsample":
            x_upsampled = F.interpolate(x, size=y.shape[-2:], mode="nearest")
            x_upsampled = x_upsampled[:, : y.shape[1], :, :]
            x_trans = x_upsampled
        return y + x_trans


class ResConv2dBlock(Conv2d):
    def forward(self, x):  # pylint: disable=arguments-renamed
        y = super().forward(x)
        y = F.relu(y)
        return x + y


class ResidualDenseBlock(nn.Sequential):
    def __init__(self, in_size, n_layers, out_size):
        layers = []
        if in_size >= out_size:
            size = in_size
        else:
            layers.append(nn.Linear(in_size, out_size))
            layers.append(nn.ReLU())
            size = out_size
            layers.append(BatchNorm1d(size))
        for _ in range(n_layers):
            layers.append(ResidualLinear(size))
            layers.append(BatchNorm1d(size))
        layers.append(nn.Linear(size, out_size))
        super().__init__(*layers)

        self.in_size = in_size
        self.out_size = out_size

    def forward(self, x):  # pylint: disable=arguments-renamed
        y = super().forward(x)
        if self.in_size == self.out_size:
            x_trans = x
        elif self.in_size < self.out_size:
            repetitions = math.ceil(y.shape[1] / x.shape[1])
            x_trans = x.repeat(1, repetitions)[:, : y.shape[1]]
        else:
            x_trans = x[:, : y.shape[1]]
        return x_trans + y


class ResidualLinear(nn.Linear):
    def __init__(self, in_size: int) -> None:
        super().__init__(in_size, in_size)

    def forward(self, x):  # pylint: disable=arguments-renamed
        y = super().forward(x)
        return x + F.relu(y)
