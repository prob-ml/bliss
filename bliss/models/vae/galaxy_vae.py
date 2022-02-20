import torch
from torch import Tensor
from torch.distributions import Normal
from torch.nn import functional as F

from bliss.models.galaxy_net import CenteredGalaxyDecoder, CenteredGalaxyEncoder, OneCenteredGalaxyAE


class OneCenteredGalaxyVAE(OneCenteredGalaxyAE):
    def make_encoder(self, slen, latent_dim, n_bands, hidden):
        return CenteredGalaxyVencoder(slen, latent_dim, n_bands, hidden)

    def make_decoder(self, slen, latent_dim, n_bands, hidden):
        return CenteredGalaxyDecoder(slen, latent_dim, n_bands, hidden, use_weight_norm=True)


class CenteredGalaxyVencoder(CenteredGalaxyEncoder):
    def __init__(self, slen, latent_dim, n_bands, hidden):
        super().__init__(slen, latent_dim * 2, n_bands, hidden, use_weight_norm=True)
        self.latent_dim = latent_dim
        self.register_buffer("zero", torch.tensor(0.0))
        self.register_buffer("one", torch.tensor(1.0))
        self.p_z = Normal(self.zero, self.one)

    def encode(self, image: Tensor):
        encoded = self.features(image)
        mean, logvar = torch.split(encoded, (self.latent_dim, self.latent_dim), -1)
        return Normal(mean, F.softplus(logvar) + 1e-3)

    def forward(self, image: Tensor):
        q_z = self.encode(image)
        z = q_z.rsample()
        log_pz = self.p_z.log_prob(z).sum(-1)
        log_qz = q_z.log_prob(z).sum(-1)
        return z, log_pz - log_qz
