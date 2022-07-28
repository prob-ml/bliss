import torch


def get_snr(img_no_background, background):
    """Calculate Signal-to-Noise Ratio.

    Args:
        background: Tensor of shape `(N x C x H x W)` containing the background for the image.
        img_no_background: Tensor of shape `(N x C x H x W)` where N is the batch_size, C is the
            number of band, H is height and W is weight, containing image data.

    Returns:
        Tensor of shape `(N,)`. The Signal-to-Noise Ratio.
    """
    assert img_no_background.shape == background.shape
    image = img_no_background + background
    return torch.sqrt(torch.sum(img_no_background**2 / image, dim=(3, 2, 1)))
