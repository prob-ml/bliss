from pathlib import Path

import numpy as np
import pytest
import torch

from bliss.catalog import FullCatalog, TileCatalog, SourceType
from bliss.surveys.decals import DecalsFullCatalog


@pytest.fixture(scope="module")
def multi_source_tilecat():
    d = {
        "n_sources": torch.tensor([[[2, 1], [0, 2]]]),
        "locs": torch.zeros(1, 2, 2, 2, 2),
        "galaxy_bools": torch.ones((1, 2, 2, 2, 1)).bool(),
        "galaxy_params": torch.zeros((1, 2, 2, 2, 7)),
        "star_fluxes": torch.zeros((1, 2, 2, 2, 1)),
    }
    d["galaxy_params"][0, 0, 0, :, 0] = torch.tensor([1000, 500])
    d["galaxy_params"][0, 0, 1, :, 0] = torch.tensor([10000, 200])
    d["galaxy_params"][0, 1, 0, :, 0] = torch.tensor([0, 800])
    d["galaxy_params"][0, 1, 1, :, 0] = torch.tensor([300, 600])

    return TileCatalog(4, d)


def test_unallowed_param():
    d_tile = {
        "n_sources": torch.zeros((1, 2, 2)),
        "locs": torch.zeros((1, 2, 2, 1, 2)),
        "unallowed": torch.zeros((1, 2, 2, 1, 2)),
    }
    with pytest.raises(ValueError):
        TileCatalog(4, d_tile)

    d_full = {
        "n_sources": torch.tensor([1]),
        "plocs": torch.tensor([5, 5]).reshape(1, 1, 2),
        "unallowed": torch.zeros(1, 1, 1),
    }
    with pytest.raises(ValueError):
        FullCatalog(10, 10, d_full)


def test_multiple_sources_one_tile():
    d = {
        "n_sources": torch.tensor([2]),
        "plocs": torch.tensor([[0.5, 0.5], [0.6, 0.6]]).reshape(1, 2, 2),
        "source_type": torch.full((1, 2, 1), SourceType.GALAXY),
    }
    full_cat = FullCatalog(2, 2, d)

    with pytest.raises(ValueError) as error_info:
        full_cat.to_tile_params(1, 1, ignore_extra_sources=False)
    assert error_info.value.args[0] == "# of sources per tile exceeds `max_sources_per_tile`."

    # should return only first source in first tile.
    tile_cat = full_cat.to_tile_params(1, 1, ignore_extra_sources=True)
    assert torch.equal(tile_cat.n_sources, torch.tensor([[[1, 0], [0, 0]]]))

    correct_locs = torch.tensor([[[0.5, 0.5], [0, 0]], [[0, 0], [0, 0]]]).reshape(1, 2, 2, 1, 2)
    assert torch.allclose(tile_cat.locs, correct_locs)

    correct_gbs = torch.tensor([[[1], [0]], [[0], [0]]]).reshape(1, 2, 2, 1, 1)
    assert torch.equal(tile_cat.galaxy_bools, correct_gbs)


def test_load_decals_from_file(cfg):
    sample_file = Path(cfg.paths.decals).joinpath("tractor-3366m010.fits")
    decals_cat = DecalsFullCatalog.from_file(sample_file)

    ras = decals_cat["ra"].numpy()
    decs = decals_cat["dec"].numpy()

    assert np.isclose(np.min(ras), 336.5, atol=1e-4)
    assert np.isclose(np.max(ras), 336.75, atol=1e-4)
    assert np.isclose(np.min(decs), -1.125, atol=1e-4)
    assert np.isclose(np.max(decs), -0.875, atol=1e-4)


def test_load_decals_ranges(cfg):
    sample_file = Path(cfg.paths.decals).joinpath("tractor-3366m010.fits")
    ra_lim = (336.6, 336.7)
    dec_lim = (-1.042, -0.92)
    decals_cat = DecalsFullCatalog.from_file(sample_file, ra_lim, dec_lim)

    ras = decals_cat["ra"].numpy()
    decs = decals_cat["dec"].numpy()

    assert np.min(ras) >= ra_lim[0]
    assert np.max(ras) <= ra_lim[1]
    assert np.min(decs) >= dec_lim[0]
    assert np.max(decs) <= dec_lim[1]


def test_restrict_tile_cat_to_brightest(multi_source_tilecat):
    cat = multi_source_tilecat.get_brightest_source_per_tile()
    assert cat.max_sources == 1
    assert cat["galaxy_params"][0, 0, 0, 0, 0] == 1000
    assert cat["galaxy_params"][0, 1, 1, 0, 0] == 600
    assert cat.n_sources.max() == 1

    # do it again to make sure nothing changes
    assert cat.get_brightest_source_per_tile().max_sources == 1


def test_filter_tile_cat_by_flux(multi_source_tilecat):
    cat = multi_source_tilecat.filter_tile_catalog_by_flux(300, 2000)
    assert cat.max_sources == 2
    assert torch.equal(cat["galaxy_params"][0, 0, 0, :, 0], torch.tensor([1000, 500]))
    assert torch.equal(cat["galaxy_params"][0, 0, 1, :, 0], torch.tensor([0, 0]))
    assert torch.equal(cat["galaxy_params"][0, 1, 0, :, 0], torch.tensor([0, 0]))
    assert torch.equal(cat["galaxy_params"][0, 1, 1, :, 0], torch.tensor([0, 600]))


def test_bin_full_cat_by_flux():
    d = {
        "n_sources": torch.tensor([3]),
        "plocs": torch.tensor([[10, 10], [20, 20], [30, 30]]).reshape(1, 3, 2),
        "galaxy_bools": torch.tensor([1, 1, 0]).reshape(1, 3, 1),
        "mags": torch.tensor([20, 23, 22]).reshape(1, 3, 1),
    }
    binned_cat = FullCatalog(40, 40, d).apply_param_bin("mags", 21, 24)
    new_plocs = binned_cat.plocs[:, 0 : binned_cat.n_sources]
    new_mags = binned_cat["mags"][:, 0 : binned_cat.n_sources]

    assert new_plocs.shape == (1, 2, 2)
    assert new_mags.max() < 24
    assert new_mags.min() > 21
