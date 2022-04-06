from typing import Tuple

import numpy as np
import torch
from einops import rearrange
from torch import Tensor, nn

from bliss.datasets.sdss import SloanDigitalSkySurvey


class ConstantBackground(nn.Module):
    def __init__(self, background: Tuple[float, ...]):
        super().__init__()
        bg: Tensor = torch.tensor(background)
        bg = rearrange(bg, "c -> 1 c 1 1")
        self.register_buffer("background", bg, persistent=False)

    def sample(self, shape) -> Tensor:
        assert isinstance(self.background, Tensor)
        batch_size, c, hlen, wlen = shape
        return self.background.expand(batch_size, c, hlen, wlen)


class SimulatedSDSSBackground(nn.Module):
    def __init__(self, sdss_dir, run, camcol, field, bands):
        super().__init__()
        sdss_data = SloanDigitalSkySurvey(
            sdss_dir=sdss_dir,
            run=run,
            camcol=camcol,
            fields=(field,),
            bands=bands,
        )
        background = torch.from_numpy(sdss_data[0]["background"])
        background = rearrange(background, "c h w -> 1 c h w", c=len(bands))
        self.register_buffer("background", background, persistent=False)
        self.height, self.width = self.background.shape[-2:]

    def sample(self, shape) -> Tensor:
        assert isinstance(self.background, Tensor)
        batch_size, c, hlen, wlen = shape
        assert self.background.shape[1] == c
        h_diff, w_diff = self.height - hlen, self.width - wlen
        h = 0 if h_diff == 0 else np.random.randint(h_diff)
        w = 0 if w_diff == 0 else np.random.randint(w_diff)
        bg = self.background[:, :, h : (h + hlen), w : (w + wlen)]
        return bg.expand(batch_size, -1, -1, -1)
