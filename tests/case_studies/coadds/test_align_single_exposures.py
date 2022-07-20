import case_studies.coadds.align_single_exposures
import galsim 
import torch

def test_galsim_align():
    pixel_scale = 0.393
    slen = 9 
    g0 = galsim.Gaussian(sigma=1.0)
    g0 = g0.shear(e1=0, e2=0)
    img0 = g0.drawImage(nx=slen, ny=slen, scale=pixel_scale)
    dithers = [((-0.5 - 0.5) * torch.rand((2,)) + 0.5).numpy() for x in range(10)]
    img = []
    for i in dithers:
        wcs1 = galsim.OffsetWCS(scale = 0.393, origin = galsim.PositionD(i))
        im = g0.drawImage(nx=slen, ny=slen, scale=pixel_scale, offset = i)
        im = im.array
        img.append(im)
    align_single_exposures(img0, img, slen, dithers)