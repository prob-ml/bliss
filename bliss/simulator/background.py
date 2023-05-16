from typing import Tuple

import numpy as np
import torch
from einops import rearrange
from torch import Tensor, nn

# ideally this module wouldn't depend on any specific survey
from bliss.surveys.sdss import SloanDigitalSkySurvey


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
    def __init__(self, sdss_dir, params):
        super().__init__()
        run = params["run"]
        camcol = params["camcol"]
        field_params = params["field"]
        if isinstance(field_params, int):
            fields = (field_params,)  # allow single integer value
        else:
            assert field_params["start"] < field_params["end"]
            fields = range(field_params["start"], field_params["end"], field_params["step"])
        bands = params["bands"]

        sdss_data = SloanDigitalSkySurvey(
            sdss_dir=sdss_dir,
            run=run,
            camcol=camcol,
            fields=fields,
            bands=bands,
        )
        background = torch.from_numpy(
            np.stack([sdss_data[i]["background"] for i in range(len(sdss_data))], axis=0)
        )
        if len(sdss_data) == 1:
            background = rearrange(background, "c h w -> 1 c h w", c=len(bands))
        self.register_buffer("background", background, persistent=False)
        self.height, self.width = self.background.shape[-2:]

    def sample(self, shape) -> Tensor:
        assert isinstance(self.background, Tensor)
        batch_size, c, hlen, wlen = shape
        assert self.background.shape[1] == c

        # select region to sample from (same for all images in batch for simplicity)
        h_diff, w_diff = self.height - hlen, self.width - wlen
        h = 0 if h_diff == 0 else np.random.randint(h_diff)
        w = 0 if w_diff == 0 else np.random.randint(w_diff)

        # sample region from random background for each image in batch
        n = np.random.randint(self.background.shape[0], size=(batch_size,))
        return self.background[n, :, h : (h + hlen), w : (w + wlen)]
