import numpy as np
import torch


def snr(img, background):
    """Calculate Signal-to-Noise Ratio.
    Args:
        img: Tensor of shape `(N x C x H x W)` where N is the batch_size, C is the
            number of band, H is height and W is weight, containing image data.
    Returns:
        Tensor of shape `(N x 1)`. The Signal-to-Noise Ratio.
    """
    assert img.shape == background.shape

    sn = torch.sqrt(torch.sum((img - background) ** 2 / (img), dim=(3, 2, 1)))

    return 1 / sn
