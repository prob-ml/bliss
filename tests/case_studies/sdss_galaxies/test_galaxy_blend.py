import torch
from hydra.utils import instantiate

from bliss.catalog import TileCatalog
from bliss.datasets.galsim_galaxies import GalsimBlends


def test_galaxy_blend(get_sdss_galaxies_config, devices):
    overrides = {
        "datasets.galsim_blended_galaxies.num_workers": 0,
        "datasets.galsim_blended_galaxies.batch_size": 4,
        "datasets.galsim_blended_galaxies.n_batches": 1,
        "datasets.galsim_blended_galaxies.prior.max_n_sources": 3,
    }
    cfg = get_sdss_galaxies_config(overrides, devices)
    blend_ds: GalsimBlends = instantiate(cfg.datasets.galsim_blended_galaxies)

    for b in blend_ds.train_dataloader():
        images, _ = b.pop("images"), b.pop("background")
        tile_cat = TileCatalog(4, b)
        full_cat = tile_cat.to_full_params()
        max_n_sources = full_cat.max_sources
        n_sources = full_cat.n_sources
        plocs = full_cat.plocs
        params = full_cat["galaxy_params"]
        snr = full_cat["snr"]
        blendedness = full_cat["blendedness"]
        ellips = full_cat["ellips"]
        mags = full_cat["mags"]
        galaxy_fluxes = full_cat["galaxy_fluxes"]
        assert images.shape == (4, 1, 88, 88)  # 40 + 24 * 2
        assert params.shape == (4, max_n_sources, 7)
        assert plocs.shape == (4, max_n_sources, 2)
        assert snr.shape == (4, max_n_sources, 1)
        assert blendedness.shape == (4, max_n_sources, 1)
        assert n_sources.shape == (4,)
        assert ellips.shape == (4, max_n_sources, 2)
        assert mags.shape == (4, max_n_sources, 1)
        assert galaxy_fluxes.shape == (4, max_n_sources, 1)

        # check empty if no sources
        for ii, n in enumerate(n_sources):
            assert torch.all(snr[ii, n:] == 0)
            assert torch.all(blendedness[ii, n:] == 0)
            assert torch.all(plocs[ii, n:] == 0)
