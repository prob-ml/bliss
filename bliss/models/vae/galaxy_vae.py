import torch
from torch import Tensor
from torch.distributions import Normal

from bliss.models.galaxy_net import CenteredGalaxyEncoder, OneCenteredGalaxyAE


class OneCenteredGalaxyVAE(OneCenteredGalaxyAE):
    def __init__(
        self,
        slen: int,
        latent_dim: int,
        hidden: int,
        n_bands: int,
        optimizer_params: dict = None,
    ):
        super().__init__(slen, latent_dim, hidden, n_bands, optimizer_params=optimizer_params)
        self.enc = CenteredGalaxyVencoder(
            slen=slen, latent_dim=latent_dim, hidden=hidden, n_bands=n_bands
        )


class CenteredGalaxyVencoder(CenteredGalaxyEncoder):
    def __init__(self, slen, latent_dim, n_bands, hidden):
        super().__init__(slen, latent_dim * 2, n_bands, hidden)
        self.latent_dim = latent_dim
        self.register_buffer("zero", torch.tensor(0.0))
        self.register_buffer("one", torch.tensor(1.0))
        self.p_z = Normal(self.zero, self.one)

    def encode(self, image: Tensor):
        encoded = self.features(image)
        mean, logvar = torch.split(encoded, (self.latent_dim, self.latent_dim), -1)
        return Normal(mean, logvar.exp() + 1e-3)

    def forward(self, image: Tensor):
        q_z = self.encode(image)
        z = q_z.rsample()
        log_pz = self.p_z.log_prob(z)
        log_qz = q_z.log_prob(z)
        return z, log_pz - log_qz
