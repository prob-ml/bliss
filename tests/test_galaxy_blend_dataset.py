import torch
from astropy.table import Table

from bliss.catalog import FullCatalog
from bliss.datasets.generate_blends import generate_dataset
from bliss.datasets.lsst import (
    get_default_lsst_psf,
    prepare_final_star_catalog,
)


def test_galaxy_blend_catalogs(home_dir):
    psf = get_default_lsst_psf()
    catsim_table = Table.read(home_dir / "data" / "small_cat.fits")
    all_star_mags = prepare_final_star_catalog()
    blends_ds = generate_dataset(100, catsim_table, all_star_mags, psf, 10)

    tile_slen = 5
    slen = 50
    bp = 24
    n_tiles = slen // tile_slen
    size = bp * 2 + tile_slen * n_tiles

    # check batches are not all the same
    assert not torch.all(blends_ds["images"][0] == blends_ds["images"][1])
    assert not torch.all(blends_ds["plocs"][0] == blends_ds["plocs"][1])

    images = blends_ds.pop("images")
    noiseless = blends_ds.pop("noiseless")
    blends_ds.pop("paddings")
    blends_ds.pop("centered_sources")
    blends_ds.pop("uncentered_sources")
    full_cat = FullCatalog(slen, slen, blends_ds)
    tile_cat = full_cat.to_tile_params(tile_slen, ignore_extra_sources=True)
    full_cat_from_tiles = tile_cat.to_full_params()
    assert images.shape == noiseless.shape

    # check tile_catalog is internally consistent
    assert tile_cat.n_sources.shape == (100, n_tiles, n_tiles)
    assert tile_cat.locs.shape == (100, n_tiles, n_tiles, 2)

    max_n_sources = full_cat.max_n_sources

    # checks on tile catalogs
    for ii in range(100):
        for jj in range(n_tiles):
            for kk in range(n_tiles):
                n = tile_cat.n_sources[ii, jj, kk].item()
                l = tile_cat.locs[ii, jj, kk]
                gb = tile_cat["galaxy_bools"][ii, jj, kk].item()
                sb = tile_cat["star_bools"][ii, jj, kk].item()
                if n > 0:
                    assert n == 1  # at most 1 source per tile
                    assert (l > 0).all()
                    assert l[0] <= 1 and l[1] <= 1
                    assert gb == 1 or sb == 1
                    assert gb == 0 or sb == 0
                else:
                    assert n == 0
                    assert l[0] == 0 and l[1] == 0
                    assert gb == 0 and sb == 0

    # checks on full catalog
    n_sources = full_cat.n_sources
    plocs = full_cat.plocs
    params = full_cat["galaxy_params"]
    gbools = full_cat["galaxy_bools"]
    sbools = full_cat["star_bools"]
    assert images.shape == (100, 1, size, size)  # 40 + 24 * 2
    assert params.shape == (100, max_n_sources, 11)  # 10 is new galaxy params from catsim
    assert plocs.shape == (100, max_n_sources, 2)
    assert n_sources.shape == (100,)
    assert gbools.shape == (100, max_n_sources, 1)
    assert max_n_sources >= n_sources.max()
    assert params[:, :, -3].max() > 300 and params[:, :, -4].max() > 300  # angles in degrees
    assert params[:, :, -3].max() <= 360 and params[:, :, -4].max() <= 360  # angles in degrees

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
    assert full_cat["galaxy_bools"].sum() >= tile_cat["galaxy_bools"].sum()
    assert full_cat["star_bools"].sum() >= tile_cat["star_bools"].sum()
    assert full_cat_from_tiles["galaxy_bools"].sum() == tile_cat["galaxy_bools"].sum()
    assert full_cat_from_tiles["star_bools"].sum() == tile_cat["star_bools"].sum()

    total_n_stars = full_cat["star_bools"].sum()
    assert total_n_stars >= 1
