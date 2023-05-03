from pathlib import Path

import numpy as np
import pytest
import torch

from bliss.catalog import FullCatalog
from bliss.surveys.decals import DecalsFullCatalog


def test_multiple_sources_one_tile():
    d = {
        "n_sources": torch.tensor([2]),
        "plocs": torch.tensor([[0.5, 0.5], [0.6, 0.6]]).reshape(1, 2, 2),
        "galaxy_bools": torch.tensor([1, 1]).reshape(1, 2, 1),
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
    assert torch.equal(tile_cat["galaxy_bools"], correct_gbs)


def test_load_decals_from_file(cfg):
    sample_file = Path(cfg.paths.decals).joinpath("tractor-3366m010.fits")
    decals_cat = DecalsFullCatalog.from_file(sample_file)

    assert not torch.all(decals_cat.plocs)  # all plocs should be 0 by default

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
