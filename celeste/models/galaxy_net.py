import inspect
import numpy as np
import pytorch_lightning as pl
from .. import device

import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
import torch.nn as nn
import torch.nn.functional as f
from torch.distributions import Normal
from torch.optim import Adam


class Flatten(nn.Module):
    @staticmethod
    def forward(tensor):
        return tensor.view(tensor.size(0), -1)


class CenteredGalaxyEncoder(nn.Module):  # recognition, inference
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
            nn.BatchNorm1d(hidden, track_running_stats=False),
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


class CenteredGalaxyDecoder(nn.Module):  # generator
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
        """

        :param z: Has shape = latent_dim.
        :return:
        """
        z = self.fc(z)

        # This dimension is the number of samples.
        z = z.view(-1, 64, self.slen // 2 + 1, self.slen // 2 + 1)
        z = self.deconv(z)
        z = z[:, :, : self.slen, : self.slen]

        # first half of the bands is now used.
        # expected number of photons has to be positive, this is why we use f.relu here.
        recon_mean = f.relu(z[:, : self.n_bands])

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
        )  # shape = (8,)
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
        slen,
        latent_dim=8,
        n_bands=1,
        lr=1e-4,
        weight_decay=1e-6,
        num_workers=2,
        tt_split=0.1,
        batch_size=64,
        logging=False,
    ):
        super(OneCenteredGalaxy, self).__init__()

        self.slen = slen  # The dimensions of the image slen * slen
        self.latent_dim = latent_dim
        self.n_bands = n_bands

        self.lr = lr
        self.weight_decay = weight_decay
        self.num_workers = num_workers
        self.tt_split = tt_split
        self.batch_size = batch_size
        self.logging = logging

        self.dataset = dataset

        self.enc = CenteredGalaxyEncoder(slen, latent_dim, n_bands)
        self.dec = CenteredGalaxyDecoder(slen, latent_dim, n_bands)

        self.register_buffer("zero", torch.zeros(1))
        self.register_buffer("one", torch.ones(1))

    def forward(self, image, background):

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
        recon_mean += background
        recon_var += background

        return recon_mean, recon_var, kl_z

    def get_loss(self, image, background):
        # NOTE: image includes background.

        # sampling images from the real distribution
        # z | x ~ decoder
        recon_mean, recon_var, kl_z = self.forward(image, background)

        # -log p(x | z), dimensions: torch.Size([ nsamples, n_bands, slen, slen])
        # assuming covariance is diagonal.
        recon_losses = -Normal(recon_mean, recon_var.sqrt()).log_prob(image)

        # image.size(0) == n_samples.
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

    # TODO: Does this work???
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
        loss = self.get_loss(image, background)
        logs = {"train_loss": loss}
        return {"loss": loss, "logs": logs}

    def validation_step(self, batch, batch_idx):
        image, background = batch["image"], batch["background"]
        loss = self.get_loss(image, background)
        logs = {"val_loss": loss}
        return {"loss": loss, "logs": logs}

    def validation_epoch_end(self, outputs):

        # TODO: does the dataloader guarantee correct size?
        avg_loss = 0.0
        for output in outputs:
            avg_loss += output["loss"]
        avg_loss /= len(outputs) * self.batch_size

        # TODO: How to save correctly into the logging folder?
        torch.save(self.vae.state_dict(), vae_file.as_posix())

        # TODO: add plotting images and their residuals with plot_reconstruction function.

        return {"val_loss": outputs[-1]["val_loss"]}

    def rmse_pp(self, image, background):
        recon_mean, recon_var, kl_z = self.forward(image, background)
        return torch.sqrt(((recon_mean - image) ** 2).sum()) / self.slen ** 2

    @classmethod
    def from_args(cls, args_dict):
        parameters = inspect.signature(cls).parameters
        filtered_dict = {
            param: value for param, value in args_dict.items() if param in parameters
        }
        return cls(**filtered_dict)

    def plot_reconstruction(self, epoch):
        num_examples = min(10, self.batch_size)

        num_cols = 3  # also look at recon_var

        plots_path = Path(self.dir_name, f"plots")
        bands_indices = [
            min(2, self.num_bands - 1)
        ]  # only i band if available, otherwise the highest band.
        plots_path.mkdir(parents=True, exist_ok=True)

        plt.ioff()
        with torch.no_grad():
            for batch_idx, data in enumerate(self.test_loader):
                image = data["image"].cuda()  # copies from cpu to gpu memory.
                background = data[
                    "background"
                ].cuda()  # not having background will be a problem.
                num_galaxies = data["num_galaxies"]
                self.vae.eval()

                recon_mean, recon_var, _ = self.vae(image, background)
                for j in bands_indices:
                    plt.figure(figsize=(5 * 3, 2 + 4 * num_examples))
                    plt.tight_layout()
                    plt.suptitle("Epoch {:d}".format(epoch))

                    for i in range(num_examples):
                        vmax1 = image[
                            i, j
                        ].max()  # we are looking at the ith sample in the jth band.
                        vmax2 = max(
                            image[i, j].max(),
                            recon_mean[i, j].max(),
                            recon_var[i, j].max(),
                        )

                        plt.subplot(num_examples, num_cols, num_cols * i + 1)
                        plt.title("image [{} galaxies]".format(num_galaxies[i]))
                        plt.imshow(image[i, j].data.cpu().numpy(), vmax=vmax1)
                        plt.colorbar()

                        plt.subplot(num_examples, num_cols, num_cols * i + 2)
                        plt.title("recon_mean")
                        plt.imshow(recon_mean[i, j].data.cpu().numpy(), vmax=vmax1)
                        plt.colorbar()

                        plt.subplot(num_examples, num_cols, num_cols * i + 3)
                        plt.title("recon_var")
                        plt.imshow(recon_var[i, j].data.cpu().numpy(), vmax=vmax2)
                        plt.colorbar()

                    plot_file = plots_path.joinpath(f"plot_{epoch}_{j}")
                    plt.savefig(plot_file.as_posix())
                    plt.close()

                break
