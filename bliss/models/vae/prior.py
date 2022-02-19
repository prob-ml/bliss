import torch

from bliss.models.vae.galaxy_flow import CenteredGalaxyLatentFlow


class GalaxyVAEPrior:
    def __init__(
        self,
        vae_flow: CenteredGalaxyLatentFlow,
        vae_flow_ckpt: str,
    ):
        self.flow = vae_flow
        self.flow.load_state_dict(torch.load(vae_flow_ckpt, map_location=vae_flow.device))

    def sample(self, n_latent_samples, device):
        self.flow = self.flow.to(device=device)
        return self.flow.sample(n_latent_samples)
