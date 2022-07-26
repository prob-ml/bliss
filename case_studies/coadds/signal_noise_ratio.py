import numpy as np
import torch


def snr(img):
    """Calculate Signal-to-Noise Ratio.
    Args:
        img: Tensor of shape `(N x C x H x W)` where N is the batch_size, C is the
            number of band, H is height and W is weight, containing image data.
    Returns:
        Tensor of shape `(N x 1)`. The Signal-to-Noise Ratio.
    """

    # sn = torch.sqrt(torch.sum((img)**2 / (img), dim=(3, 2, 1)))
    sn = torch.mean(img, dim=(3, 2, 1)) / torch.std(img, dim=(3, 2, 1))

    return sn
