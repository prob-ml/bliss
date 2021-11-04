import pytorch_lightning as pl
import torch
from nflows import transforms, distributions, flows
from nflows.transforms.base import Transform

from bliss.models.galaxy_net import OneCenteredGalaxyAE
from bliss.optimizer import get_optimizer


class CenteredGalaxyLatentFlow(pl.LightningModule):
    def __init__(
        self,
        latent_dim=64,
        optimizer_params: dict = None,
        autoencoder_ckpt=None,
        n_layers=10,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.latent_dim = latent_dim

        # Embed the autoencoder
        assert autoencoder_ckpt is not None
        autoencoder = OneCenteredGalaxyAE.load_from_checkpoint(autoencoder_ckpt)
        self.encoder = autoencoder.get_encoder()
        self.encoder.requires_grad_(False)

        self.decoder = autoencoder.get_decoder()
        self.decoder.requires_grad_(False)

        transform_list = [BatchNormTransform(latent_dim)]

        for _ in range(n_layers):
            transform_list.extend(
                [
                    transforms.MaskedAffineAutoregressiveTransform(
                        features=self.latent_dim,
                        hidden_features=64,
                    ),
                    transforms.RandomPermutation(self.latent_dim),
                ]
            )

        transform = transforms.CompositeTransform(transform_list)

        # Define a base distribution.
        base_distribution = distributions.StandardNormal(shape=[self.latent_dim])

        # Combine into a flow.
        self.flow = flows.Flow(transform=transform, distribution=base_distribution)

    def forward(self, image, background):
        latent = self.encoder(image, background)
        return -self.flow.log_prob(latent).mean()

    def training_step(self, batch, batch_idx):
        images, background = batch["images"], batch["background"]
        loss = self(images, background)
        self.log("train/loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, background = batch["images"], batch["background"]
        loss = self(images, background)
        self.log("val/loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        assert self.hparams["optimizer_params"] is not None, "Need to specify `optimizer_params`."
        name = self.hparams["optimizer_params"]["name"]
        kwargs = self.hparams["optimizer_params"]["kwargs"]
        return get_optimizer(name, self.parameters(), kwargs)


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
