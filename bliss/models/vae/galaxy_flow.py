import pytorch_lightning as pl
import torch
from matplotlib import pyplot as plt
from nflows import transforms, distributions, flows
from nflows.transforms.base import Transform

from bliss.models.vae.galaxy_net import OneCenteredGalaxyVAE
from bliss.optimizer import get_optimizer


class CenteredGalaxyLatentFlow(pl.LightningModule):
    def __init__(
        self,
        vae: OneCenteredGalaxyVAE,
        vae_ckpt: str,
        optimizer_params: dict = None,
        n_layers=10,
    ):
        super().__init__()

        self.optimizer_params = optimizer_params
        # Embed the autoencoder
        # assert vae_ckpt is not None
        vae.load_state_dict(torch.load(vae_ckpt, map_location=vae.device))
        self.encoder = vae.get_encoder()
        self.encoder.requires_grad_(False)

        self.decoder = vae.get_decoder()
        self.decoder.requires_grad_(False)

        self.latent_dim = vae.latent_dim

        self.flow_main = make_flow(self.latent_dim // 2, n_layers)
        self.flow_residual = make_flow(self.latent_dim // 2, n_layers)

    def forward(self, image, background):
        latent, _ = self.encoder(image, background)
        latent_main, latent_residual = torch.split(
            latent, (self.latent_dim // 2, self.latent_dim // 2), -1
        )
        return (
            -self.flow_main.log_prob(latent_main).mean()
            - self.flow_residual.log_prob(latent_residual).mean(),
            latent,
        )

    def training_step(self, batch, batch_idx):
        images, background = batch["images"], batch["background"]
        loss, _ = self(images, background)
        self.log("train/loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, background = batch["images"], batch["background"]
        loss, latent = self(images, background)
        self.log("val/loss", loss, prog_bar=True)
        latent_main, latent_residual = torch.split(
            latent, (self.latent_dim // 2, self.latent_dim // 2), -1
        )
        u_main = self.flow_main.transform_to_noise(latent_main)
        u_residual = self.flow_residual.transform_to_noise(latent_residual)

        return {
            "latent_main": latent_main,
            "latent_residual": latent_residual,
            "u_main": u_main,
            "u_residual": u_residual,
        }

    def validation_epoch_end(self, outputs):
        """Validation epoch end (pytorch lightning)."""

        output_tensors = {
            label: torch.cat([output[label] for output in outputs]) for label in outputs[0]
        }

        base_size = 8
        agg_posterior_before = plt.figure(figsize=(base_size, base_size))
        latent_main = output_tensors["latent_main"].cpu().detach().numpy()
        plt.scatter(latent_main[:, 0], latent_main[:, 1])

        agg_posterior_after = plt.figure(figsize=(base_size, base_size))
        u_main = output_tensors["u_main"].cpu().detach().numpy()
        plt.scatter(u_main[:, 0], u_main[:, 1])

        if self.logger:
            heading = f"Epoch:{self.current_epoch}"
            self.logger.experiment.add_figure(
                f"{heading}/Aggregate posterior (before transform)", agg_posterior_before
            )
            self.logger.experiment.add_figure(
                f"{heading}/Aggregate posterior (after transform)", agg_posterior_after
            )

    def configure_optimizers(self):
        assert self.optimizer_params is not None
        return get_optimizer(
            self.optimizer_params["name"], self.parameters(), self.optimizer_params["kwargs"]
        )


def make_flow(latent_dim, n_layers):
    transform_list = [BatchNormTransform(latent_dim)]
    for _ in range(n_layers):
        transform_list.extend(
            [
                transforms.MaskedAffineAutoregressiveTransform(
                    features=latent_dim,
                    hidden_features=64,
                ),
                transforms.RandomPermutation(latent_dim),
            ]
        )

    transform = transforms.CompositeTransform(transform_list)

    # Define a base distribution.
    base_distribution = distributions.StandardNormal(shape=[latent_dim])

    # Combine into a flow.
    return flows.Flow(transform=transform, distribution=base_distribution)


class StandardizationTransform(Transform):
    def __init__(self, mu, sigma):
        self.d = mu.size(0)
        super().__init__()
        self.register_buffer("mu", mu)
        self.register_buffer("sigma", sigma)

    def forward(self, inputs, context=None):
        U = (inputs - self.mu) / self.sigma
        log_det = -self.sigma.log().sum()
        return U, log_det

    def inverse(self, inputs, context=None):
        X = inputs * self.sigma + self.mu
        log_det = self.sigma.log().sum()
        return X, log_det


class BatchNormTransform(Transform):
    def __init__(self, d, momentum=0.1, eps=1e-5):
        super().__init__()
        self.momentum = momentum
        self.eps = eps

        self.register_buffer("running_mean", torch.zeros(d))
        self.register_buffer("running_var", torch.ones(d))

    def forward(self, inputs, context=None):
        if self.training:
            batch_mean = inputs.mean(0)
            batch_var = (inputs - batch_mean).pow(2).mean(0) + self.eps
            self.running_mean.mul_(self.momentum)
            self.running_var.mul_(self.momentum)

            self.running_mean.add_(batch_mean.data * (1 - self.momentum))
            self.running_var.add_(batch_var.data * (1 - self.momentum))
            mean = batch_mean
            var = batch_var
        else:
            mean = self.running_mean
            var = self.running_var

        U = (inputs - mean) / var.sqrt()
        log_det = -0.5 * torch.log(var).sum()
        return U, log_det

    def inverse(self, inputs, context=None):
        mean = self.running_mean
        var = self.running_var
        X = inputs * var.sqrt() + mean
        log_det = 0.5 * torch.log(var).sum()
        return X, log_det
