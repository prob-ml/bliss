from typing import Tuple

import torch
from einops import rearrange
from torch import Tensor, nn


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
