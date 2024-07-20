from pathlib import Path

import torch
from astropy.table import Table

from bliss.catalog import FullCatalog
from bliss.datasets.galsim_blends import generate_dataset
from bliss.datasets.lsst import get_default_lsst_psf
from bliss.datasets.table_utils import column_to_tensor


def test_galaxy_blend_catalogs(home_dir: Path):
    psf = get_default_lsst_psf()
    catsim_table = Table.read(home_dir / "data" / "OneDegSq.fits")
    all_star_mags = column_to_tensor(
        Table.read(home_dir / "data" / "stars_med_june2018.fits"), "i_ab"
    )
    blends_ds = generate_dataset(100, catsim_table, all_star_mags, psf, 10)

    tile_slen = 4
    slen = 40
    n_tiles = slen // tile_slen

    # check batches are not all the same
    assert not torch.all(blends_ds["images"][0] == blends_ds["images"][1])
    assert not torch.all(blends_ds["plocs"][0] == blends_ds["plocs"][1])

    images = blends_ds.pop("images")
    background = blends_ds.pop("background")
    noiseless = blends_ds.pop("noiseless")
    blends_ds.pop("paddings")
    blends_ds.pop("centered_sources")
    blends_ds.pop("uncentered_sources")
    full_cat = FullCatalog(slen, slen, blends_ds)
    tile_cat = full_cat.to_tile_params(tile_slen, ignore_extra_sources=True)
    full_cat_from_tiles = tile_cat.to_full_params()
    assert images.shape == background.shape == noiseless.shape

    # check tile_catalog is internally consistent
    assert tile_cat.n_sources.shape == (100, n_tiles, n_tiles)
    assert tile_cat.locs.shape == (100, n_tiles, n_tiles, 2)

    max_n_sources = full_cat.max_n_sources

    # checks on full catalog
    n_sources = full_cat.n_sources
    plocs = full_cat.plocs
    params = full_cat["galaxy_params"]
    gbools = full_cat["galaxy_bools"]
    sbools = full_cat["star_bools"]
    assert images.shape == (100, 1, 88, 88)  # 40 + 24 * 2
    assert params.shape == (100, max_n_sources, 10)  # 10 is new galaxy params from catsim
    assert plocs.shape == (100, max_n_sources, 2)
    assert n_sources.shape == (100,)
    assert gbools.shape == (100, max_n_sources, 1)
    assert max_n_sources >= n_sources.max()

    # TODO: checks on tiles parameters

    # plocs
    assert torch.all(plocs / slen >= 0) and torch.all(plocs / slen <= 1)

    # now more complicated comparisons
    for ii in range(100):
        n = n_sources[ii].item()

        for jj in range(max_n_sources):
            is_on = jj < n
            gbool = gbools[ii, jj].item()
            sbool = sbools[ii, jj].item()
        if is_on:
            if gbool:
                assert torch.any(params[ii, jj] > 0)
                assert not sbool
            else:
                assert sbool
                assert torch.all(params[ii, jj] == 0)
            assert torch.all(plocs[ii, jj] != 0)
        else:
            assert not gbool and not sbool
            assert torch.all(params[ii, jj] == 0)
            assert torch.all(plocs[ii, jj] == 0)
            assert torch.all(plocs[ii, jj] == 0)

        # check empty if no sources
        assert torch.all(plocs[ii, n:] == 0)

    # check consistency of tile catalog vs full catalog
    assert tile_cat.n_sources.sum() == full_cat_from_tiles.n_sources.sum()
    assert full_cat.n_sources.sum() >= tile_cat.n_sources.sum()

    total_n_stars = full_cat["star_bools"].sum()
    assert total_n_stars >= 1
