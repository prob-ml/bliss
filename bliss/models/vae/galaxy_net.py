import math

import torch
from matplotlib import pyplot as plt
from nflows.distributions import StandardNormal
from nflows.flows import Flow
from torch import nn
from torch.distributions import Normal
from torch.nn import functional as F
from torch.nn.modules.batchnorm import BatchNorm1d
from torch.nn.utils import weight_norm

from bliss.models.galaxy_net import (
    CenteredGalaxyDecoder,
    CenteredGalaxyEncoder,
    OneCenteredGalaxyAE,
)

plt.switch_backend("Agg")
plt.ioff()


class OneCenteredGalaxyVAE(OneCenteredGalaxyAE):
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
        latent_dim: int,
        n_bands: int,
        slen: int = 53,
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
        self.optimizer_params = optimizer_params

        self.main_encoder = CenteredGalaxyVAEEncoder(
            slen=slen,
            latent_dim=(latent_dim // 2) * 2,
            n_bands=n_bands,
        )
        self.main_decoder = CenteredGalaxyVAEDecoder(
            slen=slen, latent_dim=latent_dim // 2, n_bands=n_bands
        )
        self.main_autoencoder = nn.Sequential(self.main_encoder, self.main_decoder)

        self.residual_encoder = CenteredGalaxyVAEEncoder(
            slen=slen,
            latent_dim=(latent_dim // 2) * 2,
            n_bands=n_bands,
        )
        self.residual_decoder = CenteredGalaxyVAEDecoder(
            slen=slen, latent_dim=latent_dim // 2, n_bands=n_bands
        )
        self.residual_autoencoder = nn.Sequential(self.residual_encoder, self.residual_decoder)

        self.residual_delay_n_steps = residual_delay_n_steps
        assert slen == 53, "Currently slen is fixed at 53"
        self.slen = slen
        self.latent_dim = latent_dim
        self.min_sd = min_sd

        # Hack for backward compatiblity with old checkpoint
        self.register_buffer("prior_mean", torch.tensor(0.0))
        self.register_buffer("prior_var", torch.tensor(1.0))

        self.dist_main = StandardNormal([latent_dim // 2])
        self.dist_residual = StandardNormal([latent_dim // 2])

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
            min_sd=self.min_sd,
        )

    def get_decoder(self):
        return OneCenteredGalaxyDecoder(self.main_decoder, self.residual_decoder)

    def sample_latent(self, n_samples):
        latent_main = self.dist_main.sample(n_samples)
        latent_residual = self.dist_residual.sample(n_samples)
        return torch.cat((latent_main, latent_residual), dim=-1)

    def sample(self, n_samples):
        z = self.sample_latent(n_samples)
        decoder = self.get_decoder()
        return decoder(z)

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
        recon_mean_main, pq_latent_main = self._main_forward(images, background)
        loss_recon_main = self._get_likelihood_loss(images, recon_mean_main)
        loss_prior_main = -pq_latent_main.sum()
        loss_main = loss_recon_main + loss_prior_main
        self.log("val/loss_main", loss_main)

        recon_mean_residual, pq_latent_residual = self._residual_forward(images - recon_mean_main)
        recon_mean_final = F.relu(recon_mean_main + recon_mean_residual)
        loss_recon = self._get_likelihood_loss(images, recon_mean_final)
        loss_prior = loss_prior_main - pq_latent_residual.sum()
        loss = loss_recon + loss_prior
        self.log("val/loss", loss)

        # metrics
        residuals = (images - recon_mean_final) / torch.sqrt(recon_mean_final)
        residuals_main = (images - recon_mean_main) / torch.sqrt(recon_mean_main)
        recon_mean_residual = recon_mean_residual / torch.sqrt(recon_mean_main)
        self.log("val/max_residual", residuals.abs().max())

        # Aggregate posterior
        latent_main_mean, latent_main_sd = torch.split(
            self.main_encoder(images - background), (self.latent_dim // 2, self.latent_dim // 2), -1
        )
        latent_dist = Normal(latent_main_mean, F.softplus(latent_main_sd) + self.min_sd)
        latent_main = latent_dist.rsample()
        if isinstance(self.dist_main, Flow):
            latent_main = self.dist_main.transform_to_noise(latent_main)
        return {
            "images": images,
            "recon_mean_main": recon_mean_main,
            "recon_mean_residual": recon_mean_residual,
            "recon_mean": recon_mean_final,
            "residuals": residuals,
            "residuals_main": residuals_main,
            "latent_main": latent_main,
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
        latent_main = output_tensors["latent_main"].cpu().detach().numpy()
        plt.scatter(latent_main[:, 0], latent_main[:, 1])

        if self.logger:
            heading = f"Epoch:{self.current_epoch}"
            self.logger.experiment.add_figure(f"{heading}/Random Images", fig_random)
            self.logger.experiment.add_figure(f"{heading}/Worst Images", fig_worst)
            self.logger.experiment.add_figure(f"{heading}/grid_examples", grid_example)
            self.logger.experiment.add_figure(f"{heading}/Aggregate posterior", agg_posterior)

    def test_step(self, batch, batch_idx):
        """Testing step (pytorch lightning)."""
        images, background = batch["images"], batch["background"]
        recon_mean, _ = self(images, background)
        residuals = (images - recon_mean) / torch.sqrt(recon_mean)
        self.log("max_residual", residuals.abs().max())

    def _main_forward(self, image, background):
        latent_main_mean, latent_main_sd = torch.split(
            self.main_encoder(image - background), (self.latent_dim // 2, self.latent_dim // 2), -1
        )
        latent_dist = Normal(latent_main_mean, F.softplus(latent_main_sd) + self.min_sd)
        latent_main = latent_dist.rsample()
        p_latent_main = self.dist_main.log_prob(latent_main)
        q_latent_main = latent_dist.log_prob(latent_main).sum(-1)
        recon_mean_main = F.relu(self.main_decoder(latent_main)) + background
        return recon_mean_main, p_latent_main - q_latent_main

    def _residual_forward(self, residual):
        latent_residual_mean, latent_residual_sd = torch.split(
            self.residual_encoder(residual), (self.latent_dim // 2, self.latent_dim // 2), -1
        )
        latent_dist = Normal(latent_residual_mean, F.softplus(latent_residual_sd) + self.min_sd)
        latent_residual = latent_dist.rsample()
        p_latent_residual = self.dist_residual.log_prob(latent_residual)
        q_latent_residual = latent_dist.log_prob(latent_residual).sum(-1)
        recon_mean_residual = self.residual_decoder(latent_residual)
        return recon_mean_residual, p_latent_residual - q_latent_residual

    def _ensure_dist_on_device(self):
        self.dist_main = Normal(
            torch.tensor(0.0, device=self.device), torch.tensor(1.0, device=self.device)
        )
        self.dist_residual = Normal(
            torch.tensor(0.0, device=self.device), torch.tensor(1.0, device=self.device)
        )


class CenteredGalaxyVAEEncoder(nn.Module):
    def __init__(
        self, slen=53, latent_dim=8, n_bands=1, use_batch_norm=False, use_weight_norm=True
    ):

        super().__init__()

        self.slen = slen
        self.latent_dim = latent_dim
        self.conv_layer = CenteredGalaxyEncoder(
            slen=slen, latent_dim=latent_dim, n_bands=n_bands, use_weight_norm=True
        )

        kernels = [3, 3, 3, 3, 1]
        in_size = 2 ** len(kernels)
        layers = []
        if use_batch_norm:
            layers.append(nn.BatchNorm1d(in_size))
        layers.append(
            ResidualDenseBlock(
                in_size,
                3,
                latent_dim,
                use_batch_norm=use_batch_norm,
                use_weight_norm=use_weight_norm,
            )
        )
        if use_batch_norm:
            layers.append(nn.BatchNorm1d(latent_dim))
        self.dense_layer = nn.Sequential(*layers)

    def forward(self, image):
        """Encodes galaxy from image."""
        z1 = self.conv_layer(image)
        return self.dense_layer(z1)


class CenteredGalaxyVAEDecoder(nn.Module):
    def __init__(
        self, slen=53, latent_dim=8, n_bands=1, use_batch_norm=False, use_weight_norm=True
    ):
        super().__init__()
        kernels = [3, 3, 3, 3, 1]
        in_size = 2 ** len(kernels)
        self.conv_layer = CenteredGalaxyDecoder(
            slen=slen, latent_dim=latent_dim, n_bands=n_bands, use_weight_norm=True
        )
        self.dense_layer = ResidualDenseBlock(
            latent_dim, 3, in_size, use_batch_norm=use_batch_norm, use_weight_norm=use_weight_norm
        )

    def forward(self, z):
        """Decodes image from latent representation."""
        z1 = self.dense_layer(z)
        return self.conv_layer(z1)


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
        main_encoder: CenteredGalaxyVAEEncoder,
        main_decoder: CenteredGalaxyVAEDecoder,
        residual_encoder: CenteredGalaxyVAEEncoder,
        slen: int = None,
        allow_pad: bool = False,
        min_sd: float = 1e-3,
    ):
        """Initializer.

        Args:
            main_encoder: The main encoder
            main_decoder: The main decoder
            residual_encoder: The residual encoder
            slen: (optional) The side-length of the galaxy image. Only needed if allow_pad is True.
            allow_pad: (optional) Should padding be added to an image
                if its size is less than slen? Defaults to False.
            min_sd: Minimum sd to use in encoded distribution
        """
        super().__init__()
        self.main_encoder = main_encoder
        self.main_decoder = main_decoder
        self.residual_encoder = residual_encoder
        self.slen = slen
        self.allow_pad = allow_pad
        self.min_sd = min_sd

        self.register_buffer("prior_mean", torch.tensor(0.0))
        self.register_buffer("prior_scale", torch.tensor(1.0))
        self.prior_main = Normal(self.prior_mean, self.prior_scale)
        self.prior_residual = Normal(self.prior_mean, self.prior_scale)

    def forward(self, image, background=0):
        assert image.shape[-2] == image.shape[-1]
        if self.allow_pad:
            if image.shape[-1] < self.slen:
                d = self.slen - image.shape[-1]
                lpad = d // 2
                upad = d - lpad
                min_val = image.min().item()
                image = F.pad(image, (lpad, upad, lpad, upad), value=min_val)
                if isinstance(background, torch.Tensor):
                    background = F.pad(background, (lpad, upad, lpad, upad))
        latent_main_params = self.main_encoder(image - background)
        d_main = latent_main_params.shape[-1] // 2
        latent_main_mean, latent_main_sd = torch.split(latent_main_params, (d_main, d_main), -1)
        latent_dist = Normal(latent_main_mean, F.softplus(latent_main_sd) + self.min_sd)
        latent_main = latent_dist.rsample()
        q_latent_main = latent_dist.log_prob(latent_main)
        p_latent_main = self.prior_main.log_prob(latent_main)
        pq_latent_main = p_latent_main - q_latent_main

        recon_mean_main = F.relu(self.main_decoder(latent_main)) + background
        latent_residual_params = self.residual_encoder(image - recon_mean_main)
        d_residual = latent_residual_params.shape[-1] // 2
        latent_residual_mean, latent_residual_sd = torch.split(
            latent_residual_params, (d_residual, d_residual), -1
        )

        latent_residual_dist = Normal(
            latent_residual_mean, F.softplus(latent_residual_sd) + self.min_sd
        )
        latent_residual = latent_residual_dist.rsample()
        q_latent_residual = latent_residual_dist.log_prob(latent_residual)
        p_latent_residual = self.prior_residual.log_prob(latent_residual)
        pq_latent_residual = p_latent_residual - q_latent_residual

        pq_latent = pq_latent_main.sum(-1) + pq_latent_residual.sum(-1)

        return (torch.cat((latent_main, latent_residual), dim=-1), pq_latent)


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


class ResidualDenseBlock(nn.Sequential):
    def __init__(self, in_size, n_layers, out_size, use_batch_norm=False, use_weight_norm=True):
        if use_weight_norm:
            wn = weight_norm
        else:
            wn = lambda x: x
        layers = []
        if in_size >= out_size:
            size = in_size
        else:
            layers.append(wn(nn.Linear(in_size, out_size)))
            layers.append(nn.ReLU())
            size = out_size
            if use_batch_norm:
                layers.append(BatchNorm1d(size))
        for _ in range(n_layers):
            layers.append(wn(ResidualLinear(size)))
            if use_batch_norm:
                layers.append(BatchNorm1d(size))
        layers.append(wn(nn.Linear(size, out_size)))
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
