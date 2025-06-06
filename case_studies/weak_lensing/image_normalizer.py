import torch
from einops import rearrange


class NanNormalizer(torch.nn.Module):
    def num_channels_per_band(self):
        return 1

    def get_input_tensor(self, batch):
        return rearrange(
            torch.nan_to_num(batch["images"], nan=batch["images"].nanmedian()),
            "b bands h w -> b bands 1 h w",
        )
