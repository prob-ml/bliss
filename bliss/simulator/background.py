from typing import Dict, Tuple

import numpy as np
import torch
from einops import rearrange
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
from torch import Tensor, nn

# ideally this module wouldn't depend on any specific survey
from bliss.surveys.sdss import SloanDigitalSkySurvey


class ConstantBackground(nn.Module):
    def __init__(self, background: Tuple[float, ...], **_kwargs):
        super().__init__()
        bg: Tensor = torch.tensor([background])
        bg = rearrange(bg[0], "c -> 1 c 1 1")
        self.register_buffer("background", bg, persistent=False)

    def sample(self, shape, **_kwargs) -> Tensor:
        assert isinstance(self.background, Tensor)
        batch_size, c, hlen, wlen = shape
        return self.background.expand(batch_size, c, hlen, wlen)


class SimulatedSDSSBackground(nn.Module):
    def __init__(self, sdss_fields: DictConfig):
        super().__init__()
        sdss_dir = sdss_fields["dir"]
        field_list = sdss_fields["field_list"]
        bands = sdss_fields["bands"]  # use same bands across fields

        # Add all backgrounds from specified run/col/field to list
        backgrounds = []
        for param_obj in field_list:
            params: Dict = OmegaConf.to_container(param_obj)
            sdss = SloanDigitalSkySurvey(sdss_dir=sdss_dir, bands=bands, **params)
            backgrounds.extend([field["background"] for field in sdss])

        background = torch.from_numpy(np.stack(backgrounds, axis=0))

        self.register_buffer("background", background, persistent=False)
        self.height, self.width = self.background.shape[-2:]

    def sample(self, shape, rcf_indices) -> Tensor:
        """Sample a random region to use as the background for each image.

        Args:
            shape: shape of background to return (channels x height x width)
            rcf_indices: batch_size-length array of indices to index into background list

        Returns:
            Tensor: batch_size x shape tensor of backgrounds
        """
        assert isinstance(self.background, Tensor)
        batch_size, c, hlen, wlen = shape
        assert self.background.shape[1] == c
        assert rcf_indices.shape[0] == batch_size

        # select region to sample from (same for all images in batch for simplicity)
        h_diff, w_diff = self.height - hlen, self.width - wlen
        h = 0 if h_diff == 0 else np.random.randint(h_diff)
        w = 0 if w_diff == 0 else np.random.randint(w_diff)

        # sample region from specified background for each image in batch
        return self.background[rcf_indices, :, h : (h + hlen), w : (w + wlen)]
