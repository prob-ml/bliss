import numpy as np
import torch
from hydra.utils import instantiate


def test_galaxy_blend(get_sdss_galaxies_config, devices):
    overrides = {
        "datasets.galsim_blended_galaxies.batch_size": 4,
        "datasets.galsim_blended_galaxies.n_batches": 1,
        "datasets.galsim_blended_galaxies.max_n_sources": 3,
    }
    cfg = get_sdss_galaxies_config(overrides, devices)
    blend_ds = instantiate(cfg.datasets.galsim_blended_galaxies)

    for b in blend_ds.train_dataloader():
        images = b["images"]
        noiseless = b["noiseless"]
        single_images_center = b["individual_noiseless_centered"]
        single_images_uncenter = b["individual_noiseless_uncentered"]
        params = b["params"]
        n_sources = b["n_sources"]
        plocs = b["plocs"]
        snr = b["snr"]
        blendedness = b["blendedness"]
        ellips = b["ellips"]
        mags = b["mags"]
        fluxes = b["fluxes"]
        assert images.shape == (4, 1, 81, 81)
        assert noiseless.shape == (4, 1, 81, 81)
        assert single_images_center.shape == (4, 3, 1, 81, 81)
        assert single_images_uncenter.shape == (4, 3, 1, 81, 81)
        assert params.shape == (4, 3, 7)
        assert plocs.shape == (4, 3, 2)
        assert snr.shape == (4, 3)
        assert blendedness.shape == (4, 3)
        assert n_sources.shape == (4, 1)
        assert ellips.shape == (4, 3, 2)
        assert mags.shape == (4, 3)
        assert fluxes.shape == (4, 3)

        # check empty if no sources
        for ii, n in enumerate(n_sources):
            assert torch.all(snr[ii, n:] == 0)
            assert torch.all(blendedness[ii, n:] == 0)
            assert torch.all(plocs[ii, n:] == 0)
            assert torch.all(single_images_center[ii, n:] == 0)
            assert torch.all(single_images_uncenter[ii, n:] == 0)

        # check flux matches in noiseless and single noiseless images.
        for ii in range(images.shape[0]):
            flux1 = noiseless[ii].sum().item()
            flux2 = single_images_center[ii].sum().item()
            flux3 = single_images_uncenter[ii].sum().item()
            np.testing.assert_allclose(flux1, flux2, atol=1e4, rtol=1e-3)
            np.testing.assert_allclose(flux2, flux3, atol=1e4, rtol=1e-3)
            np.testing.assert_allclose(flux1, flux3, atol=1e4, rtol=1e-3)
