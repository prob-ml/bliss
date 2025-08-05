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

class NanClipNormalizer(torch.nn.Module):
    def __init__(self, clip_min=None, clip_max=None, clip_percentile=99):
        super().__init__()
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.clip_percentile = clip_percentile
    
    def num_channels_per_band(self):
        return 1

    def get_input_tensor(self, batch):
        images = batch["images"]
        
        # Replace NaNs first
        images_clean = torch.nan_to_num(images, nan=images.nanmedian())

        if self.clip_max is None:
            clip_max = torch.quantile(images_clean, self.clip_percentile / 100)
        else:
            clip_max = self.clip_max
        
        # Clip to fixed bounds
        images_clipped = torch.clamp(images_clean, self.clip_min, clip_max)
        
        return rearrange(
            images_clipped,
            "b bands h w -> b bands 1 h w",
        )
