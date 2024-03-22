from typing import Tuple

import torch
from einops import rearrange
from torch import Tensor


def get_constant_background(value: float, shape: Tuple[int, ...]):
    b, c, h, w = shape
    bg = torch.tensor([value])
    bg = rearrange(bg, "1 -> 1 1 1 1")
    return bg.expand(b, c, h, w)


def add_noise_and_background(image: Tensor, background: Tensor) -> Tensor:
    image_with_background = image + background
    noise = image_with_background.sqrt() * torch.randn_like(image_with_background)
    return image_with_background + noise
