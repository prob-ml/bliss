import torch
from hydra.utils import instantiate

from bliss.catalog import FullCatalog, TileCatalog
from bliss.datasets.blends import GalsimBlends


def test_galaxy_blend_catalogs(get_galsim_galaxies_config):
    cfg = get_galsim_galaxies_config({})
    blend_ds: GalsimBlends = instantiate(cfg.blends_datasets.blends)
    slen = blend_ds.slen
    tile_slen = blend_ds.tile_slen
    n_tiles = slen // tile_slen

    # check batches are not all the same
    assert not torch.all(blend_ds[0]["images"] == blend_ds[1]["images"])
    assert not torch.all(blend_ds[0]["full_params"]["plocs"] == blend_ds[1]["full_params"]["plocs"])

    total_n_stars = 0
    for ii in range(100):
        b = blend_ds[ii]
        images, _ = b.pop("images"), b.pop("background")
        tile_cat = TileCatalog(tile_slen, b["tile_params"])
        full_cat = FullCatalog(slen, slen, b["full_params"])
        full_cat_from_tiles = tile_cat.to_full_params()

        # check tile_catalog is internally consistent
        assert tile_cat.n_sources.shape == (1, n_tiles, n_tiles)
        assert tile_cat.locs.shape == (1, n_tiles, n_tiles, 1, 2)

        # check full catalogs are correct
        for cat in (full_cat, full_cat_from_tiles):
            max_n_sources = cat.max_sources
            n_sources = cat.n_sources
            plocs = cat.plocs
            params = cat["galaxy_params"]
            snr = cat["snr"]
            blendedness = cat["blendedness"]
            ellips = cat["ellips"]
            mags = cat["mags"]
            gbools = cat["galaxy_bools"]
            star_fluxes = cat["star_fluxes"]
            star_log_fluxes = cat["star_log_fluxes"]
            star_bools = cat["star_bools"]
            assert images.shape == (1, 1, 88, 88)  # 40 + 24 * 2
            assert params.shape == (1, max_n_sources, 10)  # 10 is new galaxy params from catsim
            assert plocs.shape == (1, max_n_sources, 2)
            assert snr.shape == (1, max_n_sources, 1)
            assert blendedness.shape == (1, max_n_sources, 1)
            assert n_sources.shape == (1,)
            assert ellips.shape == (1, max_n_sources, 2)
            assert mags.shape == (1, max_n_sources, 1)
            assert gbools.shape == (1, max_n_sources, 1)
            assert star_fluxes.shape == (1, max_n_sources, 1)
            assert star_log_fluxes.shape == (1, max_n_sources, 1)

            assert max_n_sources >= n_sources.max()

            # plocs
            assert torch.all(plocs / slen >= 0) and torch.all(plocs / slen <= 1)

            # now more complicated comparisons
            n_sources = n_sources.item()
            for ii in range(max_n_sources):
                is_on = ii < n_sources
                gbool = cat["galaxy_bools"][0, ii, 0].item()
                sbool = cat["star_bools"][0, ii, 0].item()
                if is_on:
                    if gbool:
                        assert torch.any(params[0, ii] > 0)
                        assert star_fluxes[0, ii].item() == 0
                        assert star_log_fluxes[0, ii].item() == 0
                        assert not sbool
                        assert snr[0, ii, 0].item() > 0
                    else:
                        assert sbool
                        assert torch.all(params[0, ii] == 0)
                        assert star_fluxes[0, ii].item() > 0
                        assert star_log_fluxes[0, ii].item() > 0
                        assert snr[0, ii, 0].item() > 0
                        total_n_stars += 1
                    assert torch.all(plocs[0, ii] != 0)
                else:
                    assert not gbool and not sbool
                    assert star_fluxes[0, ii].item() == 0
                    assert star_log_fluxes[0, ii].item() == 0
                    assert torch.all(params[0, ii] == 0)
                    assert torch.all(plocs[0, ii] == 0)
                    assert snr[0, ii, 0].item() == 0
                    assert torch.all(plocs[0, ii] == 0)

            # check empty if no sources
            assert torch.all(snr[0, n_sources:, 0] == 0)
            assert torch.all(blendedness[0, n_sources:, 0] == 0)
            assert torch.all(plocs[0, n_sources:] == 0)

        # check consistency of tile catalog vs full catalog
        assert tile_cat.n_sources.sum() == full_cat_from_tiles.n_sources.sum()
        assert full_cat.n_sources.sum() >= tile_cat.n_sources.sum()

    assert total_n_stars >= 1


def check_blend_images(get_galsim_galaxies_config):
    """We want to test in case there is a small miscentering effect."""
    pass
