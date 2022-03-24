from typing import Optional

import torch

from bliss.models.vae.galaxy_flow import CenteredGalaxyLatentFlow


class GalaxyVAEPrior:
    def __init__(
        self,
        min_flux: float,
        max_flux: float,
        alpha: float,
        latent_dim: int,
        vae_flow: Optional[CenteredGalaxyLatentFlow] = None,
        vae_flow_ckpt: str = None,
    ):
        self.min_flux = min_flux
        self.max_flux = max_flux
        self.alpha = alpha
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
        fluxes = self._draw_pareto_flux(samples.shape[0], device).unsqueeze(-1)
        log_fluxes = fluxes.log()
        samples = torch.cat((samples, log_fluxes), dim = -1)
        return samples

    def _draw_pareto_flux(self, n, device):
        # draw pareto conditioned on being less than f_max
        u_max = 1.0 - (self.min_flux / self.max_flux) ** self.alpha
        uniform_samples = torch.rand(n, device=device) * u_max
        return self.min_flux / (1.0 - uniform_samples) ** (1 / self.alpha)