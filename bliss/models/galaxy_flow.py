import pytorch_lightning as pl
from nflows import transforms, distributions, flows

from bliss.models.galaxy_net import OneCenteredGalaxyAE
from bliss.optimizer import get_optimizer


class CenteredGalaxyLatentFlow(pl.LightningModule):
    def __init__(
        self,
        latent_dim=64,
        optimizer_params: dict = None,
        autoencoder_ckpt=None,
        n_layers=10,
        latents_file=None,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.latent_dim = latent_dim

        transform_list = []

        for _ in range(n_layers):
            transform_list.extend(
                [
                    transforms.MaskedAffineAutoregressiveTransform(
                        features=self.latent_dim,
                        hidden_features=64,
                    ),
                ]
            )

        transform = transforms.CompositeTransform(transform_list)

        # Define a base distribution.
        base_distribution = distributions.StandardNormal(shape=[self.latent_dim])

        # Combine into a flow.
        self.flow = flows.Flow(transform=transform, distribution=base_distribution)

        # Embed the autoencoder
        assert autoencoder_ckpt is not None
        autoencoder = OneCenteredGalaxyAE.load_from_checkpoint(autoencoder_ckpt)
        self.encoder = autoencoder.get_encoder()
        self.encoder.requires_grad_(False)

        self.decoder = autoencoder.get_decoder()
        self.decoder.requires_grad_(False)

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
