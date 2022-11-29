from typing import Tuple

import torch
from torch import Tensor
from torch.distributions import Normal
from torch.nn import functional as F

from bliss.models.galaxy_net import (
    CenteredGalaxyDecoder,
    CenteredGalaxyEncoder,
    OneCenteredGalaxyAE,
)


class OneCenteredGalaxyVAE(OneCenteredGalaxyAE):
    def make_encoder(self, slen, latent_dim, n_bands, hidden):
        return CenteredGalaxyVencoder(slen, latent_dim, n_bands, hidden)

    def make_decoder(self, slen, latent_dim, n_bands, hidden):
        return CenteredGalaxyDecoder(slen, latent_dim, n_bands, hidden, use_weight_norm=True)

    def make_deblender(self, slen, latent_dim, n_bands, hidden):
        return CenteredGalaxyVencoder(slen, latent_dim, n_bands, hidden)


class CenteredGalaxyVencoder(CenteredGalaxyEncoder):
    def __init__(self, slen, latent_dim, n_bands, hidden):
        super().__init__(slen, latent_dim * 2, n_bands, hidden, use_weight_norm=True)
        self.latent_dim = latent_dim
        self.register_buffer("zero", torch.tensor(0.0))
        self.register_buffer("one", torch.tensor(1.0))
        self.p_z = Normal(self.zero, self.one)

    def sample(self, image: Tensor, deterministic=False) -> Tensor:
        q_z = self._encode(image)
        if not deterministic:
            z = q_z.rsample()
        else:
            z = q_z.loc
        return z

    def forward(self, image: Tensor) -> Tuple[Tensor, Tensor]:
        q_z = self._encode(image)
        z = q_z.rsample()
        p_z = Normal(self.p_z.loc, self.p_z.scale)
        log_pz = p_z.log_prob(z).sum(-1)
        assert isinstance(log_pz, Tensor)
        assert not torch.any(torch.isnan(log_pz))
        assert not torch.any(torch.isinf(log_pz))

        log_qz = q_z.log_prob(z).sum(-1)
        assert isinstance(log_qz, Tensor)
        assert not torch.any(torch.isnan(log_qz))
        assert not torch.any(torch.isinf(log_qz))

        pq_divergence = log_pz - log_qz
        return z, pq_divergence

    def _encode(self, image: Tensor) -> Normal:
        encoded = self.features(image)
        mean, logvar = torch.split(encoded, [self.latent_dim, self.latent_dim], -1)
        return Normal(mean, F.softplus(logvar) + 1e-3)
