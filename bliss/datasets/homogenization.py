import numpy as np
import torch
import matplotlib as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from astropy.visualization import astropy_mpl_style
plt.style.use(astropy_mpl_style)

from bliss.inference import SDSSFrame

def N_S_Ratio(img, background):
    """Calculate Noise-to-Signal Ratio

    Args:
        img: Tensor of shape `(N x C x H x W)` where N is the batch_size,
            C is the number of band, H is height and W is weight. The image data.
        backgroud: Integer.

    Returns:
        NSR: Tensor of shape `(N x 1)`. The Noise-to-Signal Ratio.
    """

    Ip = img - background
    SN = torch.sqrt(torch.sum(Ip ** 2 / (Ip + background)))
    NSR = 1 / SN
    
    return NSR

def wiener_filter(img, psf, K=10):
    """Apply Wiener filter on images

    Args:
        img: Tensor of shape `(N x C x H x W)` where N is the batch_size,
            C is the number of band, H is height and W is weight. The convloved
            image data.
        psf: Tensor of shape `(N x C x H x W)`. The Point Spread Function.
        K: Integer Noise-to-Signal Ratio.

    Returns:
        m_map : Tensor of shape `(N x C x H x W)`. The deconvolved image data.
    """
    
    img2 = torch.clone(img)
    img_fft = torch.fft.fft2(img2)
    psf_fft = torch.fft.fft2(psf)
    _, _, m, n = img2.shape
    
    laps = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
    m1 = (m - 3) // 2
    n1 = (n - 3) // 2
    laps = np.pad(laps, [[m1, m - m1 - 3], [n1, n - n1 - 3]])
    laps = torch.from_numpy(laps)
    laps_fft = torch.fft.fft2(laps)

    M = torch.conj(psf_fft)/(torch.abs(psf_fft) ** 2 + K * torch.abs(laps_fft) **2)
    m = M * img_fft
    m_map = torch.fft.fftshift(torch.fft.ifft2(m).real)

    return m_map

def psf_homo(img, psf, new_psf, K=10):
    """Homogenize the PSF in astronomical images

    Args:
        img: Tensor of shape `(N x C x H x W)`, where N is the batch_size,
            C is the number of bands, H is height and W is weight. The convloved
            image data.
        psf: Tensor of shape `(N x C x H1 x W1)`. The Point Spread Function(PSF) used
            for deconvolution.
        new_psf: Tensor of shape `(N x C x H2 x W2)`. The PSF used for convolution.
        K: Integer of Noise-to-Signal Ratio

    Return: 
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
        padding = (m, w-m-psf.shape[3], n, h-n-psf.shape[2])
        psf_pad = torch.nn.functional.pad(psf, padding)
        assert psf_pad.shape[2] == h
        assert psf_pad.shape[3] == w
    else:
        psf_pad = psf

    # Deconvolution with Wiener filter
    decv = wiener_filter(img, psf_pad, K)

    # Padding new PSFs to let them have same height and weight as image
    if not (h == new_psf.shape[2] and w == new_psf.shape[3]):
        m = (w - new_psf.shape[3]) // 2
        n = (h - new_psf.shape[2]) // 2
        padding = (m, w-m-new_psf.shape[3], n, h-n-new_psf.shape[2])
        new_psf_pad = torch.nn.functional.pad(new_psf, padding)
        assert new_psf_pad.shape[2] == h
        assert new_psf_pad.shape[3] == w
    else:
        new_psf_pad = new_psf

    # Convoution with new PSFs
    decv_fft = torch.fft.fft2(decv)
    psf_fft = torch.fft.fft2(new_psf_pad)
    Mixing_fft = decv_fft * psf_fft
    Mixing = torch.fft.fftshift(torch.fft.ifft2(Mixing_fft).real)

    return Mixing, decv

# load sdss data
sdss_dir = 'bliss/data/sdss/'
pixel_scale = 0.393
coadd_file = "bliss/data/coadd_catalog_94_1_12.fits"
frame = SDSSFrame(sdss_dir, pixel_scale, coadd_file)
background = frame.background
image = frame.image

print(background)