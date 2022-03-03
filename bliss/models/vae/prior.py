from typing import Optional

import torch

from bliss.models.vae.galaxy_flow import CenteredGalaxyLatentFlow


class GalaxyVAEPrior:
    def __init__(
        self,
        latent_dim: int,
        vae_flow: Optional[CenteredGalaxyLatentFlow] = None,
        vae_flow_ckpt: str = None,
    ):
        self.latent_dim = latent_dim
        if vae_flow is None:
            self.flow = None
        else:
            self.flow = vae_flow
            assert self.latent_dim == self.flow.latent_dim
            self.flow.eval()
            self.flow.requires_grad_(False)
            self.flow.load_state_dict(torch.load(vae_flow_ckpt, map_location=vae_flow.device))

    def sample(self, n_latent_samples, device):
        if self.flow is None:
            samples = torch.randn((n_latent_samples, self.latent_dim), device=device)
        else:
            self.flow = self.flow.to(device=device)
            samples = self.flow.sample(n_latent_samples)
        return samples
