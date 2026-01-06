import pytorch_lightning as pl
from einops import reduce
from torch import Tensor, nn
from torch.distributions import Normal
from torch.nn.functional import relu
from torch.optim import Adam

from bliss.datasets.lsst import BACKGROUND


class OneCenteredGalaxyAE(pl.LightningModule):
    def __init__(
        self,
        slen: int = 53,  # only 53, 66, 71, 89, 98... are available
        latent_dim: int = 8,
        hidden: int = 256,
        n_bands: int = 1,
        lr: float = 1e-5,
    ):
        super().__init__()

        self.enc = self.make_encoder(slen, latent_dim, n_bands, hidden)
        self.dec = self.make_decoder(slen, latent_dim, n_bands, hidden)
        self.latent_dim = latent_dim
        self.lr = lr
        self.register_buffer("background_sqrt", BACKGROUND.sqrt())

    def make_encoder(
        self, slen: int, latent_dim: int, n_bands: int, hidden: int
    ) -> "CenteredGalaxyEncoder":
        return CenteredGalaxyEncoder(slen, latent_dim, n_bands, hidden)

    def make_decoder(
        self, slen: int, latent_dim: int, n_bands: int, hidden: int
    ) -> "CenteredGalaxyDecoder":
        return CenteredGalaxyDecoder(slen, latent_dim, n_bands, hidden)

    def forward(self, images) -> Tensor:
        """Gets reconstructed images from running through encoder and decoder."""
        z = self.enc(images)
        return self.dec(z)

    def get_loss(self, images: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        recon_mean: Tensor = self(images)
        log_prob_pp = -Normal(recon_mean, self.background_sqrt).log_prob(images)
        loss = log_prob_pp.sum()
        loss_avg = log_prob_pp.mean()  # might be useful to compare with different batch sizes.
        return loss, loss_avg, recon_mean

    def training_step(self, batch: dict[str, Tensor], batch_idx: int):
        """Training step (pytorch lightning)."""
        images = batch["images"]
        loss, loss_avg, _ = self.get_loss(images)
        self.log("train/loss", loss, on_step=False, on_epoch=True)
        self.log("train/loss_avg", loss_avg, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch: dict[str, Tensor], batch_idx: int):
        """Validation step (pytorch lightning)."""
        images = batch["images"]
        loss, loss_avg, recon_mean = self.get_loss(images)

        # max std. residual across all images
        res = (images - recon_mean) / self.background_sqrt
        mean_max_residual = reduce(res.abs(), "b c h w -> b", "max").mean()

        self.log("val/loss", loss)
        self.log("val/loss_avg", loss_avg)
        self.log("val/mean_max_residual", mean_max_residual)
        self.log("val/max_residual", res.abs().max())
        return loss

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.lr)


class CenteredGalaxyEncoder(nn.Module):
    """Encodes single galaxies with noise but no background."""

    def __init__(
        self,
        slen: int,
        latent_dim: int,
        n_bands: int,
        hidden: int,
    ):
        super().__init__()

        self.slen = slen
        self.latent_dim = latent_dim

        min_slen = _conv2d_out_dim(_conv2d_out_dim(slen))

        self.features = nn.Sequential(
            nn.Conv2d(n_bands, 4, 5, stride=3, padding=0),
            nn.LeakyReLU(),
            nn.Conv2d(4, 8, 5, stride=3, padding=0),
            nn.LeakyReLU(),
            nn.Flatten(1, -1),
            nn.Linear(8 * min_slen**2, hidden),
            nn.LeakyReLU(),
            nn.Linear(hidden, latent_dim),
        )

    def forward(self, image) -> Tensor:
        """Encodes galaxy from image."""
        return self.features(image)


class CenteredGalaxyDecoder(nn.Module):
    """Reconstructs noiseless galaxies from encoding with no background."""

    def __init__(self, slen=53, latent_dim=8, n_bands=1, hidden=256):
        super().__init__()

        self.slen = slen
        self.n_bands = n_bands
        self.min_slen = _conv2d_out_dim(_conv2d_out_dim(slen))
        self._validate_slen()

        self.fc = nn.Sequential(
            nn.Linear(latent_dim, hidden),
            nn.LeakyReLU(),
            nn.Linear(hidden, 8 * self.min_slen**2),
            nn.LeakyReLU(),
        )

        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(8, 4, 5, stride=3),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(4, n_bands, 5, stride=3),
        )

    def forward(self, z) -> Tensor:
        """Decodes image from latent representation."""
        z1 = self.fc(z)
        z2 = z1.view(-1, 8, self.min_slen, self.min_slen)
        z3 = self.deconv(z2)
        z4 = z3[:, :, : self.slen, : self.slen]
        assert z4.shape[-1] == self.slen and z4.shape[-2] == self.slen
        return relu(z4)

    def _validate_slen(self) -> None:
        slen2 = _conv2d_inv_dim(_conv2d_inv_dim(self.min_slen))
        if slen2 != self.slen:
            raise ValueError(f"The input slen '{self.slen}' is invalid.")


def _conv2d_out_dim(x: int) -> int:
    """Function to figure out dimension of our Conv2D."""
    return (x - 5) // 3 + 1


def _conv2d_inv_dim(x: int) -> int:
    return (x - 1) * 3 + 5
