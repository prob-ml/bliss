import galsim
import torch

from case_studies.coadds.align_single_exposures import align_single_exposures


def test_galsim_align():
    pixel_scale = 0.393
    slen = 9
    g0 = galsim.Gaussian(sigma=1.0)
    g0 = g0.shear(e1=0, e2=0)
    img0 = g0.drawImage(nx=slen, ny=slen, scale=pixel_scale, bandpass=None)
    img0 = torch.from_numpy(img0.array)
    dithers = torch.distributions.uniform.Uniform(-0.5, 0.5).sample([10, 2])
    img = []
    for i in dithers:
        im = g0.drawImage(nx=slen, ny=slen, scale=pixel_scale, offset=i, bandpass=None)
        im = im.array
        img.append(im)
    img = torch.tensor(img)
    assert align_single_exposures(img0, img, slen, dithers).shape == torch.Size(
        [len(dithers), slen - 2, slen - 2]
    )
