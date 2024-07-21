import torch
from einops import rearrange


class NullNormalizer(torch.nn.Module):
    def num_channels_per_band(self):
        return 1

    def get_input_tensor(self, batch):
        return rearrange((batch["images"] + 0.5).clamp(1e-6) * 100, "b bands h w -> b bands 1 h w")
