import numpy as np
import torch


def n_s_ratio(img, background):
    """Calculate Noise-to-Signal Ratio.

    Args:
        img: Tensor of shape `(N x C x H x W)` where N is the batch_size, C is the \
            number of band, H is height and W is weight, containing image data.
        background: Tensor of shape `(N x C x H x W)`, containing background of image.

    Returns:
        Tensor of shape `(N x 1)`. The Noise-to-Signal Ratio.
    """
    assert img.shape == background.shape

    ip = img - background
    sn = torch.sqrt(torch.sum(ip**2 / (ip + background), dim=(3, 2, 1)))

    return 1 / sn


def wiener_filter(img, psf, k):
    """Apply Wiener filter on images.

    Args:
        img: Tensor of image of shape `(N x C x H x W)` where N is the batch_size, \
            C is the number of band, H is height and W is weight, containing the image data.
        psf: Tensor of shape `(N x C x H x W)`, representing the Point Spread Function.
        k: Tensor of shape `(N x 1)`, representing the Noise-to-Signal Ratio.

    Returns:
        Tensor of shape `(N x C x H x W)`. The deconvolved image data.
    """

    img2 = torch.clone(img)
    img_fft = torch.fft.fft2(img2)
    psf_fft = torch.fft.fft2(psf)
    batch_size, _, m, n = img2.shape

    laps = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
    m1 = (m - 3) // 2
    n1 = (n - 3) // 2
    laps = np.pad(laps, [[m1, m - m1 - 3], [n1, n - n1 - 3]])
    laps = torch.from_numpy(laps)
    laps_fft = torch.fft.fft2(laps)

    k = k.reshape(batch_size, 1, 1, 1)
    f = torch.conj(psf_fft) / (torch.abs(psf_fft) ** 2 + k * torch.abs(laps_fft) ** 2)
    m = f * img_fft
    return torch.fft.fftshift(torch.fft.ifft2(m).real)


def psf_homo(img, psf, new_psf, background):
    """Homogenize the PSF in astronomical images.

    Args:
        img: Tensor of shape `(N x C x H x W)`, where N is the batch_size, C is the \
        number of bands, H is height and W is weight, containing the convloved image data.
        psf: Tensor of shape `(N x C x H1 x W1)`, representing the Point Spread Function(PSF) used \
            for deconvolution.
        new_psf: Tensor of shape `(N x C x H2 x W2)`, representing the PSF used for convolution.
        background: Tensor of shape `(N x C x H x W)`,containing the backgroud of image data.

    Returns:
        Mixing: Tensor of shape `(N x C x H x W)`. The convolved image data.
        decv : Tensor of shape `(N x C x H x W)`. The deconvolved image data.
    """

    assert len(img.shape) == 4
    assert len(psf.shape) == 4
    assert len(new_psf.shape) == 4

    batch_size, _, h, w = img.shape

    assert psf.shape[0] == batch_size
    assert new_psf.shape[0] == batch_size

    # Padding PSFs to let them have same height and weight as image.
    if not (h == psf.shape[2] and w == psf.shape[3]):
        m = (w - psf.shape[3]) // 2
        n = (h - psf.shape[2]) // 2
        padding = (m, w - m - psf.shape[3], n, h - n - psf.shape[2])
        psf_pad = torch.nn.functional.pad(psf, padding)
        assert psf_pad.shape[2] == h
        assert psf_pad.shape[3] == w
    else:
        psf_pad = psf

    # Calculate NSR
    k = n_s_ratio(img, background)

    # Deconvolution with Wiener filter
    decv = wiener_filter(img, psf_pad, k)

    # Padding new PSFs to let them have same height and weight as image
    if not (h == new_psf.shape[2] and w == new_psf.shape[3]):
        m = (w - new_psf.shape[3]) // 2
        n = (h - new_psf.shape[2]) // 2
        padding = (m, w - m - new_psf.shape[3], n, h - n - new_psf.shape[2])
        new_psf_pad = torch.nn.functional.pad(new_psf, padding)
        assert new_psf_pad.shape[2] == h
        assert new_psf_pad.shape[3] == w
    else:
        new_psf_pad = new_psf

    # Convoution with new PSFs
    decv_fft = torch.fft.fft2(decv)
    psf_fft = torch.fft.fft2(new_psf_pad)
    mixing_fft = decv_fft * psf_fft
    mixing = torch.fft.fftshift(torch.fft.ifft2(mixing_fft).real)

    return mixing, decv
