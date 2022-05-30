from pathlib import Path

import galsim
import numpy as np
import torch

from bliss.inference import SDSSFrame
from case_studies.psf_homogenization.homogenization import psf_homo


def _check_flux(img, psf, new_psf, background):
    m, m_or = psf_homo(img, psf, new_psf, background)

    assert m.shape == img.shape
    assert m_or.shape == img.shape

    orig_sum = img[0][0].sum()
    conv_sum1 = m[0][0].sum()
    conv_sum2 = m[1][0].sum()

    assert torch.abs(orig_sum - conv_sum1) < 0.01 * orig_sum
    assert torch.abs(orig_sum - conv_sum2) < 0.01 * orig_sum


def test_homo_sdssframe(get_config, devices):
    pixel_scale = 0.393
    cfg = get_config({}, devices)
    sdss_dir = Path(cfg.paths.sdss)
    coadd_file = Path(cfg.paths.data).joinpath("coadd_catalog_94_1_12.fits")
    frame = SDSSFrame(sdss_dir, pixel_scale, coadd_file)
    background = frame.background
    image = frame.image

    psf = image[0][0][447:460, 1157:1170]
    psf = psf - psf.min()
    psf /= psf.sum()

    big_psf = galsim.Gaussian(fwhm=7.0).drawImage(nx=50, ny=50, scale=pixel_scale).array
    small_psf = galsim.Gaussian(fwhm=2.0).drawImage(nx=50, ny=50, scale=pixel_scale).array
    big_psf = torch.from_numpy(big_psf)
    small_psf = torch.from_numpy(small_psf)

    psf = psf.reshape(1, 1, 13, 13)
    big_psf = big_psf.reshape(1, 1, 50, 50)
    small_psf = small_psf.reshape(1, 1, 50, 50)

    img = torch.cat((image, image), 0)
    psf = torch.cat((psf, psf), 0)
    new_psf = torch.cat((big_psf, small_psf), 0)
    background = torch.cat((background, background), 0)

    _check_flux(img, psf, new_psf, background)


def test_homo_one_galaxy():
    background = np.ones((101, 101)) * 1000
    g = galsim.Gaussian(sigma=0.9, flux=1e5)
    psf = galsim.Gaussian(fwhm=0.7)
    pixel_scale = 0.2
    conv_gal = galsim.Convolve(g, psf)
    image = conv_gal.drawImage(nx=101, ny=101, scale=pixel_scale, bandpass=None).array + background
    image += np.random.randn(*image.shape) * np.sqrt(image)

    orig_psf = galsim.Gaussian(fwhm=0.7).drawImage(nx=101, ny=101, scale=pixel_scale).array
    big_psf = galsim.Gaussian(fwhm=7.0).drawImage(nx=101, ny=101, scale=pixel_scale).array
    small_psf = galsim.Gaussian(fwhm=2.0).drawImage(nx=101, ny=101, scale=pixel_scale).array

    image = image.astype(np.float32)
    image = torch.from_numpy(image)
    psf = torch.from_numpy(orig_psf)
    big_psf = torch.from_numpy(big_psf)
    small_psf = torch.from_numpy(small_psf)
    background = torch.from_numpy(background)

    image = image.reshape(1, 1, 101, 101)
    psf = psf.reshape(1, 1, 101, 101)
    big_psf = big_psf.reshape(1, 1, 101, 101)
    small_psf = small_psf.reshape(1, 1, 101, 101)
    background = background.reshape(1, 1, 101, 101)

    img = torch.cat((image, image), 0)
    psf = torch.cat((psf, psf), 0)
    new_psf = torch.cat((big_psf, small_psf), 0)
    background = torch.cat((background, background), 0)

    _check_flux(img, psf, new_psf, background)
