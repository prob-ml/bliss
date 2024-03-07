from typing import Tuple

import torch
from einops import rearrange


def get_constant_background(value: float, shape: Tuple[int, ...]):
    b, c, h, w = shape
    bg = torch.tensor(value)
    bg = rearrange(bg, "c -> 1 c 1 1")
    return bg.expand(b, c, h, w)
