import case_studies.coadds.align_single_exposures

def test_galsim_align():
    pixel_scale = 0.393
    slen = 9 
    g0 = galsim.Gaussian(sigma=1.0)
    g0 = g0.shear(e1=0, e2=0)
    img = g0.drawImage(nx=slen, ny=slen, scale=pixel_scale)
    dithers = [((-0.5 - 0.5) * torch.rand((2,)) + 0.5).numpy() for x in range(10)]
    align_images = case_studies.coadds.align_single_exposures.align_single_exposures(img, slen, dithers)

    assert torch.tensor(align_images).shape == torch.Size([len(dithers), slen-2, slen-2])
