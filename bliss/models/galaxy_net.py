import inspect

import numpy as np
import pytorch_lightning as pl
import matplotlib.pyplot as plt


import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.optim import Adam

from .. import device

plt.switch_backend("Agg")


class Flatten(nn.Module):
    @staticmethod
    def forward(tensor):
        return tensor.view(tensor.size(0), -1)


class CenteredGalaxyEncoder(nn.Module):
    def __init__(self, slen, latent_dim, n_bands, hidden=256):
        super(CenteredGalaxyEncoder, self).__init__()

        self.slen = slen
        self.latent_dim = latent_dim
        self.n_bands = n_bands

        self.features = nn.Sequential(
            nn.Conv2d(self.n_bands, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.ReLU(),
            Flatten(),
            nn.Linear(16 * self.slen ** 2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )

        self.fc_mean = nn.Linear(hidden, self.latent_dim)
        self.fc_var = nn.Linear(hidden, self.latent_dim)

    def forward(self, subimage):
        """
        1e-4 here is to avoid NaNs, .exp gives you positive and variance increase quickly.
        Exp is better matched for logs. (trial and error, but makes big difference)
        :param subimage: image to be encoded.
        :return:
        """
        z = self.features(subimage)
        z_mean = self.fc_mean(z)
        z_var = 1e-4 + torch.exp(self.fc_var(z))
        return z_mean, z_var


class CenteredGalaxyDecoder(nn.Module):
    def __init__(self, slen, latent_dim, n_bands, hidden=256):
        super(CenteredGalaxyDecoder, self).__init__()

        self.slen = slen  # side-length.
        self.latent_dim = latent_dim
        self.n_bands = n_bands

        self.fc = nn.Sequential(
            nn.Linear(latent_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 64 * (slen // 2 + 1) ** 2),  # shrink dimensions
            nn.ReLU(),
        )

        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(
                64, 64, 3, padding=0, stride=2
            ),  # this will increase size back to twice.
            nn.ConvTranspose2d(
                64, 2 * self.n_bands, 3, padding=0
            ),  # why channels=2 * num bands?
        )

    def forward(self, z):
        z = self.fc(z)  # latent variable

        # This dimension is the number of samples.
        z = z.view(-1, 64, self.slen // 2 + 1, self.slen // 2 + 1)
        z = self.deconv(z)
        z = z[:, :, : self.slen, : self.slen]

        # first half of the bands is now used.
        # expected number of photons has to be positive, this is why we use f.relu here.
        recon_mean = F.relu(z[:, : self.n_bands])

        # sometimes nn can get variance to be really small, if sigma gets really small
        # then small learning
        # this is what the 1e-4 is for.
        # We also want var >= mean because of the poisson noise, which is also imposed here.
        var_multiplier = 1 + 10 * torch.sigmoid(z[:, self.n_bands : (2 * self.n_bands)])
        recon_var = 1e-4 + var_multiplier * recon_mean

        # reconstructed mean and variance, these are per pixel.
        return recon_mean, recon_var

    def get_sample(self, num_samples):
        p_z = Normal(
            torch.zeros(1, dtype=torch.float, device=device),
            torch.ones(1, dtype=torch.float, device=device),
        )
        z = p_z.rsample(torch.tensor([num_samples, self.latent_dim])).view(
            num_samples, -1
        )
        samples, _ = self.forward(z)

        # samples shape = (num_samples, n_bands, slen, slen)
        return z, samples


class OneCenteredGalaxy(pl.LightningModule):

    # ---------------
    # Model
    # ----------------

    def __init__(
        self,
        dataset,
        slen=51,
        latent_dim=8,
        n_bands=1,
        num_workers=2,
        batch_size=64,
        tt_split=0.1,
        lr=1e-4,
        weight_decay=1e-6,
    ):
        super(OneCenteredGalaxy, self).__init__()

        self.slen = slen
        self.latent_dim = latent_dim
        self.n_bands = n_bands

        self.lr = lr
        self.weight_decay = weight_decay
        self.num_workers = num_workers
        self.tt_split = tt_split
        self.batch_size = batch_size

        self.dataset = dataset

        self.enc = CenteredGalaxyEncoder(slen, latent_dim, n_bands)
        self.dec = CenteredGalaxyDecoder(slen, latent_dim, n_bands)

        self.register_buffer("zero", torch.zeros(1))
        self.register_buffer("one", torch.ones(1))

    @pl.core.decorators.auto_move_data
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
        recon_mean, recon_var = self.dec.forward(z)

        # kl can behave wildly w/out background.
        recon_mean = recon_mean + background
        recon_var = recon_var + background

        return recon_mean, recon_var, kl_z

    @staticmethod
    def get_loss(image, recon_mean, recon_var, kl_z):
        # NOTE: image includes background.

        # -log p(x | z), dimensions: torch.Size([ nsamples, n_bands, slen, slen])
        # assuming covariance is diagonal.
        recon_losses = -Normal(recon_mean, recon_var.sqrt()).log_prob(image)

        # image.size(0) = n_samples.
        recon_losses = recon_losses.view(image.size(0), -1).sum(1)

        # ELBO
        loss = (recon_losses + kl_z).sum()

        return loss

    # ---------------
    # Data
    # ----------------

    def train_dataloader(self):
        split = len(self.dataset) * self.tt_split
        train_indices = np.arange(split, len(self.dataset))
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            sampler=SubsetRandomSampler(train_indices),
        )

    def val_dataloader(self):

        test_indices = np.arange(len(self.dataset) * self.tt_split)
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            sampler=SubsetRandomSampler(test_indices),
        )

    # ---------------
    # Optimizer
    # ----------------

    def configure_optimizers(self):
        return Adam(
            [{"params": self.parameters(), "lr": self.lr}],
            weight_decay=self.weight_decay,
        )

    # ---------------
    # Training
    # ----------------

    def training_step(self, batch, batch_idx):
        image, background = batch["image"], batch["background"]
        recon_mean, recon_var, kl_z = self(image, background)
        loss = self.get_loss(image, recon_mean, recon_var, kl_z)
        logs = {"train_loss": loss}
        return {"loss": loss, "log": logs}

    def validation_step(self, batch, batch_idx):
        image, background = batch["image"], batch["background"]
        recon_mean, recon_var, kl_z = self(image, background)
        loss = self.get_loss(image, recon_mean, recon_var, kl_z)
        return {
            "val_loss": loss,
            "image": image,
            "recon_mean": recon_mean,
            "recon_var": recon_var,
        }

    def validation_epoch_end(self, outputs):
        # logs loss, saves checkpoints (implicitly) and saves images.
        # TODO: add plotting images and their residuals with plot_reconstruction function.

        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        tensorboard_logs = {"avg_val_loss": avg_loss}

        # get first 10 images, reconstruction means, variances. Add them as a grid.
        image = outputs[-1]["image"][:5]
        recon_mean = outputs[-1]["recon_mean"][:5]
        recon_var = outputs[-1]["recon_var"][:5]

        fig = self.plot_reconstruction(image, recon_mean, recon_var)

        if self.logger:
            self.logger.experiment.add_figure(f"Images {self.current_epoch}", fig)

        return {"val_loss": avg_loss, "log": tensorboard_logs}

    # def rmse_pp(self, image, background):
    #     recon_mean, recon_var, kl_z = self.forward(image, background)
    #     return torch.sqrt(((recon_mean - image) ** 2).sum()) / self.slen ** 2

    def plot_reconstruction(self, image, recon_mean, recon_var):
        assert image.size(0) >= 5
        num_examples = 5
        num_cols = 4
        # only i band if available, otherwise the highest band.
        band_idx = min(2, self.n_bands - 1)
        residuals = (image - recon_mean) / torch.sqrt(image)
        plt.ioff()

        fig = plt.figure(figsize=(20, 20))
        plt.tight_layout()
        plt.suptitle("Epoch {:d}".format(self.current_epoch))

        for i in range(num_examples):

            plt.subplot(num_examples, num_cols, num_cols * i + 1)
            plt.title("image")
            plt.imshow(image[i, band_idx].data.cpu().numpy())
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

        return fig

    @classmethod
    def from_args(cls, dataset, args):
        args_dict = vars(args)

        parameters = list(inspect.signature(cls).parameters)
        parameters.remove("dataset")

        args_dict = {param: args_dict[param] for param in parameters}
        return cls(dataset, **args_dict)

    @staticmethod
    def add_args(parser):
        parser.add_argument("--latent-dim", type=int, default=8, help="latent dim")
        parser.add_argument(
            "--tt-split", type=float, default=0.1, help="train/test split"
        )
