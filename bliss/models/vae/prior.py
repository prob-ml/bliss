import torch

from bliss.models.vae.galaxy_flow import CenteredGalaxyLatentFlow
from bliss.models.vae.galaxy_net import OneCenteredGalaxyVAE


class GalaxyVAEPrior:
    def __init__(
        self,
        vae: OneCenteredGalaxyVAE,
        vae_ckpt: str,
        vae_flow: CenteredGalaxyLatentFlow,
        vae_flow_ckpt: str,
    ):
        vae.load_state_dict(torch.load(vae_ckpt, map_location=vae.device))
        self.vae = vae
        vae_flow.load_state_dict(torch.load(vae_flow_ckpt, map_location=vae_flow.device))
        self.vae.dist_main = vae_flow.flow_main
        self.vae.dist_residual = vae_flow.flow_residual

    def sample(self, n_latent_samples, device):
        self.vae = self.vae.to(device=device)
        return self.vae.sample_latent(n_latent_samples)
