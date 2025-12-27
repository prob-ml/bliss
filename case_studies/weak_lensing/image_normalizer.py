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


class PsfAsImage(torch.nn.Module):
    def __init__(self, num_psf_params):
        super().__init__()
        self.num_psf_params = num_psf_params

    def num_channels_per_band(self):
        """Number of input channels for model based on this input normalizer."""
        return self.num_psf_params

    def get_input_tensor(self, batch):
        assert "psf_params" in batch, "PsfAsImage specified but psf params not provided"
        n, c, h, w = batch["images"].shape
        psf_params = batch["psf_params"]
        psf_params = psf_params.view(n, c, self.num_psf_params, 1, 1)
        return psf_params.expand(n, c, self.num_psf_params, h, w)
