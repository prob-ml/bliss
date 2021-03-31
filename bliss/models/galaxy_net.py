import pytorch_lightning as pl
import matplotlib.pyplot as plt
from omegaconf import DictConfig

import torch
from torch.distributions import Normal

from bliss.optimizer import get_optimizer
from bliss.models.vae_layers import get_enc_dec

plt.switch_backend("Agg")


class OneCenteredGalaxy(pl.LightningModule):

    # ---------------
    # Model
    # ----------------

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters(cfg)

        self.enc, self.dec = get_enc_dec(cfg.model.enc_dec_name, cfg.model.params)

        self.warm_up = cfg.model.warm_up
        self.beta = cfg.model.beta

        assert self.warm_up == 0 or self.beta == 1, "Only one of 'warm_up'/'beta' can be active."

        self.register_buffer("zero", torch.zeros(1))
        self.register_buffer("one", torch.ones(1))

    def forward(self, image, background):
        # sampling images from the real distribution
        # z | x ~ decoder

        # shape = [nsamples, latent_dim]
        z_mean, z_var = self.enc.forward(image - background)

        q_z = Normal(z_mean, z_var.sqrt())
        z = q_z.rsample()

        log_q_z = q_z.log_prob(z).sum(1)
        p_z = Normal(self.zero, self.one)  # prior on z.
        log_p_z = p_z.log_prob(z).sum(1)
        kl_z = log_q_z - log_p_z  # log q(z | x) - log p(z)

        # reconstructed mean/variances images (per pixel quantities)
        # no background
        recon_mean, recon_var = self.dec.forward(z)

        # kl can behave wildly w/out background.
        recon_mean = recon_mean + background
        recon_var = recon_var + background

        return recon_mean, recon_var, kl_z

    def get_loss(self, image, recon_mean, recon_var, kl_z):
        # use a "deterministic warm-up" scheme to see if we get better results
        # on realistic galaxies. It involves "turning on" the prior penalty term slowly.
        # see: https://arxiv.org/pdf/1602.02282.pdf
        pwr = max(min(-self.warm_up + self.current_epoch, 0), -6)
        pr_penalty = self.beta * kl_z * 10 ** pwr

        # return ELBO
        # NOTE: image includes background.
        # Covariance is diagonal in latent variables.
        # recon_loss = -log p(x | z), shape: torch.Size([ nsamples, n_bands, slen, slen])
        recon_losses = -Normal(recon_mean, recon_var.sqrt()).log_prob(image)
        recon_losses = recon_losses.view(image.size(0), -1).sum(1)
        loss = (recon_losses + pr_penalty).sum()

        return loss

    # ---------------
    # Optimizer
    # ----------------

    def configure_optimizers(self):
        name = self.hparams.optimizer.name
        optim_params = self.hparams.optimizer.params
        return get_optimizer(name, self.parameters(), optim_params)

    # ---------------
    # Training
    # ----------------

    def training_step(self, batch, batch_idx):  # pylint: disable=unused-argument
        images, background = batch["images"], batch["background"]
        recon_mean, recon_var, kl_z = self(images, background)
        loss = self.get_loss(images, recon_mean, recon_var, kl_z)
        self.log("train_loss", loss)
        return loss

    # ---------------
    # Validation
    # ----------------

    def validation_step(self, batch, batch_idx):  # pylint: disable=unused-argument
        images, background = batch["images"], batch["background"]
        recon_mean, recon_var, kl_z = self(images, background)
        loss = self.get_loss(images, recon_mean, recon_var, kl_z)

        # metrics
        self.log("val_loss", loss)
        residuals = (images - recon_mean) / torch.sqrt(images)
        self.log("val_max_residual", residuals.abs().max())
        return {"images": images, "recon_mean": recon_mean, "recon_var": recon_var}

    def validation_epoch_end(self, outputs):
        images = outputs[0]["images"][:10]
        recon_mean = outputs[0]["recon_mean"][:10]
        recon_var = outputs[0]["recon_var"][:10]
        fig = self.plot_reconstruction(images, recon_mean, recon_var)
        if self.logger:
            self.logger.experiment.add_figure(f"Images {self.current_epoch}", fig)

    def plot_reconstruction(self, images, recon_mean, recon_var):
        # only plot i band if available, otherwise the highest band given.
        assert images.size(0) >= 10
        assert self.enc.n_bands == self.dec.n_bands
        n_bands = self.enc.n_bands
        num_examples = 10
        num_cols = 4
        band_idx = min(2, n_bands - 1)
        residuals = (images - recon_mean) / torch.sqrt(images)
        plt.ioff()

        fig = plt.figure(figsize=(10, 25))
        plt.suptitle("Epoch {:d}".format(self.current_epoch))

        for i in range(num_examples):

            plt.subplot(num_examples, num_cols, num_cols * i + 1)
            plt.title("images")
            plt.imshow(images[i, band_idx].data.cpu().numpy())
            plt.colorbar()

            plt.subplot(num_examples, num_cols, num_cols * i + 2)
            plt.title("recon_mean")
            plt.imshow(recon_mean[i, band_idx].data.cpu().numpy())
            plt.colorbar()

            plt.subplot(num_examples, num_cols, num_cols * i + 3)
            plt.title("recon_var")
            plt.imshow(recon_var[i, band_idx].data.cpu().numpy())
            plt.colorbar()

            plt.subplot(num_examples, num_cols, num_cols * i + 4)
            plt.title("residuals")
            plt.imshow(residuals[i, band_idx].data.cpu().numpy())
            plt.colorbar()

        plt.tight_layout()

        return fig

    def test_step(self, batch, batch_idx):  # pylint: disable=unused-argument
        images, background = batch["images"], batch["background"]
        recon_mean, _, _ = self(images, background)
        residuals = (images - recon_mean) / torch.sqrt(images)
        self.log("max_residual", residuals.abs().max())
