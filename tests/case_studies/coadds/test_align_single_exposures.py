import galsim
import torch

from case_studies.coadds.align_single_exposures import align_single_exposures


def test_galsim_align():
    pixel_scale = 0.393
    slen = 9
    g0 = galsim.Gaussian(sigma=1.0)
    g0 = g0.shear(e1=0, e2=0)
    img0 = g0.drawImage(nx=slen, ny=slen, scale=pixel_scale, bandpass=None)
    dithers = [((-0.5 - 0.5) * torch.rand((2,)) + 0.5).numpy() for x in range(10)]
    img = []
    for i in dithers:
        im = g0.drawImage(nx=slen, ny=slen, scale=pixel_scale, offset=i, bandpass=None)
        im = im.array
        img.append(im)
    align_single_exposures(img0, img, slen, pixel_scale, dithers)
