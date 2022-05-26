import galsim
import torch

from bliss.inference import SDSSFrame
from case_studies.psf_homogenization.homogenization import psf_homo

sdss_dir = "bliss/data/sdss/"
pixel_scale = 0.393
coadd_file = "bliss/data/coadd_catalog_94_1_12.fits"
frame = SDSSFrame(sdss_dir, pixel_scale, coadd_file)
background = frame.background
image = frame.image

psf = image[447:460, 1157:1170]
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

m, m_or = psf_homo(img, psf, new_psf, background)
print(m.shape)
