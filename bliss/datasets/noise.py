import torch
from torch import Tensor

from bliss.datasets.lsst import BACKGROUND


def add_noise(image: Tensor) -> Tensor:
    return image + BACKGROUND.sqrt() * torch.randn_like(image)
