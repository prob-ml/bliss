from typing import Optional, Tuple

import pytorch_lightning as pl
import torch
from matplotlib import pyplot as plt
from torch import Tensor, nn
from torch.distributions import Normal
from torch.nn import functional as F
from torch.nn.utils import weight_norm
from torch.optim import Adam
from tqdm import tqdm

from bliss.plotting import plot_image

plt.switch_backend("Agg")
plt.ioff()


def _conv2d_out_dim(x):
    """Function to figure out dimension of our Conv2D."""
    return (x - 5) // 3 + 1


def _conv2d_inv_dim(x):
    return (x - 1) * 3 + 5


def _identity(x):
    return x


class CenteredGalaxyEncoder(nn.Module):
    def __init__(self, slen, latent_dim, n_bands, hidden, use_weight_norm=False):
        super().__init__()

        self.slen = slen
        self.latent_dim = latent_dim

        min_slen = _conv2d_out_dim(_conv2d_out_dim(slen))
        wn = weight_norm if use_weight_norm else _identity

        self.features = nn.Sequential(
            wn(nn.Conv2d(n_bands, 4, 5, stride=3, padding=0)),
            nn.LeakyReLU(),
            wn(nn.Conv2d(4, 8, 5, stride=3, padding=0)),
            nn.LeakyReLU(),
            nn.Flatten(1, -1),
            wn(nn.Linear(8 * min_slen**2, hidden)),
            nn.LeakyReLU(),
            wn(nn.Linear(hidden, latent_dim)),
        )

    def sample(self, image: Tensor, deterministic=True):
        assert deterministic, "CenteredGalaxyEncoder is deterministic"
        return self.features(image)

    def forward(self, image):
        """Encodes galaxy from image."""
        z = self.sample(image)
        return z, torch.tensor(0.0, device=image.device)


class CenteredGalaxyDecoder(nn.Module):
    def __init__(self, slen=53, latent_dim=8, n_bands=1, hidden=256, use_weight_norm=False):
        super().__init__()

        self.slen = slen
        self.min_slen = _conv2d_out_dim(_conv2d_out_dim(slen))
        self._validate_slen()
        wn = weight_norm if use_weight_norm else _identity

        self.fc = nn.Sequential(
            wn(nn.Linear(latent_dim, hidden)),
            nn.LeakyReLU(),
            wn(nn.Linear(hidden, 8 * self.min_slen**2)),
            nn.LeakyReLU(),
        )

        self.deconv = nn.Sequential(
            wn(nn.ConvTranspose2d(8, 4, 5, stride=3)),
            nn.LeakyReLU(),
            wn(nn.ConvTranspose2d(4, n_bands, 5, stride=3)),
        )

    def forward(self, z):
        """Decodes image from latent representation."""
        z = self.fc(z)
        z = z.view(-1, 8, self.min_slen, self.min_slen)
        z = self.deconv(z)
        z = z[:, :, : self.slen, : self.slen]
        assert z.shape[-1] == self.slen and z.shape[-2] == self.slen
        return F.relu(z)

    def _validate_slen(self):
        slen2 = _conv2d_inv_dim(_conv2d_inv_dim(self.min_slen))
        if slen2 != self.slen:
            raise ValueError(f"The input slen '{self.slen}' is invalid.")


class OneCenteredGalaxyAE(pl.LightningModule):

    # ---------------
    # Model
    # ----------------

    def __init__(
        self,
        slen: int,
        latent_dim: int,
        hidden: int,
        n_bands: int,
        optimizer_params: Optional[dict] = None,
        ckpt: Optional[str] = None,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.optimizer_params = optimizer_params

        self.enc = self.make_encoder(slen, latent_dim, n_bands, hidden)
        self.dec = self.make_decoder(slen, latent_dim, n_bands, hidden)
        self.latent_dim = latent_dim

        self.register_buffer("zero", torch.zeros(1))
        self.register_buffer("one", torch.ones(1))

        if ckpt is not None:
            self.load_state_dict(torch.load(ckpt, map_location=self.device))  # type: ignore

    def forward(self, image, background):
        return self.reconstruct(image, background)

    def reconstruct(self, image: Tensor, background: Tensor) -> Tuple[Tensor, Tensor]:
        """Gets reconstructed image from running through encoder and decoder."""
        z, pq_z = self.enc.forward(image - background)
        recon_mean = self.dec.forward(z)
        return recon_mean + background, pq_z

    def get_loss(self, image: Tensor, recon_mean: Tensor, pq_z: Tensor):
        recon_loss = -Normal(recon_mean, recon_mean.sqrt()).log_prob(image).sum()
        pq_z_loss = -pq_z.sum()
        return recon_loss + pq_z_loss

    def make_encoder(self, slen, latent_dim, n_bands, hidden):
        return CenteredGalaxyEncoder(slen, latent_dim, n_bands, hidden)

    def make_deblender(self, slen, latent_dim, n_bands, hidden):
        return self.make_encoder(slen, latent_dim, n_bands, hidden)

    def make_decoder(self, slen, latent_dim, n_bands, hidden):
        return CenteredGalaxyDecoder(slen, latent_dim, n_bands, hidden)

    def get_encoder(self):
        return self.enc

    def get_decoder(self):
        return self.dec

    def generate_latents(self, dataloader, n_batches):
        """Induces a latent distribution for a non-probabilistic autoencoder."""
        latent_list = []
        with torch.no_grad():
            for _ in tqdm(range(n_batches)):
                galaxy = next(iter(dataloader))
                img = (galaxy["images"] - galaxy["background"]).to(self.device)
                latent_batch = self.enc(img)[0]
                latent_list.append(latent_batch)
        return torch.cat(latent_list, dim=0)

    # ---------------
    # Optimizer
    # ----------------

    def configure_optimizers(self):
        """Configures optimizers for training (pytorch lightning)."""
        return Adam(self.parameters(), **self.optimizer_params)

    # ---------------
    # Training
    # ----------------

    def training_step(self, batch, batch_idx):
        """Training step (pytorch lightning)."""
        images, background = batch["images"], batch["background"]
        recon_mean, pq_z = self.forward(images, background)
        loss = self.get_loss(images, recon_mean, pq_z)
        self.log("train/loss", loss)
        return loss

    # ----------------
    # Validation
    # ----------------

    def validation_step(self, batch, batch_idx):
        """Validation step (pytorch lightning)."""
        images, background = batch["images"], batch["background"]
        recon_mean, pq_z = self.forward(images, background)
        residuals = (images - recon_mean) / torch.sqrt(recon_mean)
        loss = self.get_loss(images, recon_mean, pq_z)

        # metrics
        self.log("val/loss", loss)
        self.log("val/max_residual", residuals.abs().max())
        return {"images": images, "recon_mean": recon_mean, "residuals": residuals}

    def validation_epoch_end(self, outputs):
        """Validation epoch end (pytorch lightning)."""

        # combine all images and recon_mean's into a single tensor
        images = torch.cat([output["images"] for output in outputs])
        recon_mean = torch.cat([output["recon_mean"] for output in outputs])
        residuals = torch.cat([output["residuals"] for output in outputs])

        fig_random = self.plot_reconstruction(images, recon_mean, residuals, mode="random")
        fig_worst = self.plot_reconstruction(images, recon_mean, residuals, mode="worst")
        if self.logger:
            self.logger.experiment.add_figure(f"Random Images {self.current_epoch}", fig_random)
            self.logger.experiment.add_figure(f"Worst Images {self.current_epoch}", fig_worst)

    def plot_reconstruction(
        self, images, recon_mean, residuals, n_examples=10, mode="random", width=10, pad=6.0
    ):
        # only plot i band if available, otherwise the highest band given.
        assert images.size(0) >= n_examples
        assert images.shape[1] == recon_mean.shape[1] == residuals.shape[1] == 1, "1 band only."
        figsize = (width, width * n_examples / 3)
        fig, axes = plt.subplots(nrows=n_examples, ncols=3, figsize=figsize)

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

            # only add titles to the first axes.
            if i == 0:
                ax_true.set_title("Images $x$", pad=pad)
                ax_recon.set_title(r"Reconstruction $\tilde{x}$", pad=pad)
                ax_res.set_title(
                    r"Residual $\left(x - \tilde{x}\right) / \sqrt{\tilde{x}}$", pad=pad
                )

            # standarize ranges of true and reconstruction
            image = images[idx, 0].detach().cpu().numpy()
            recon = recon_mean[idx, 0].detach().cpu().numpy()
            residual = residuals[idx, 0].detach().cpu().numpy()
            vmin = min(image.min().item(), recon.min().item())
            vmax = max(image.max().item(), recon.max().item())

            # plot images
            plot_image(fig, ax_true, image, vrange=(vmin, vmax))
            plot_image(fig, ax_recon, recon, vrange=(vmin, vmax))
            plot_image(fig, ax_res, residual, vrange=(vmin_res, vmax_res))

        plt.tight_layout()

        return fig

    def test_step(self, batch, batch_idx):
        """Testing step (pytorch lightning)."""
        images, background = batch["images"], batch["background"]
        recon_mean, _ = self.forward(images, background)
        residuals = (images - recon_mean) / torch.sqrt(images)
        self.log("max_residual", residuals.abs().max())
