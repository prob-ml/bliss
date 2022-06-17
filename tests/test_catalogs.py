import pytest
import torch

from bliss.catalog import FullCatalog


def test_multiple_sources_one_tile():
    d = {
        "n_sources": torch.tensor([2]),
        "plocs": torch.tensor([[0.5, 0.5], [0.6, 0.6]]).reshape(1, 2, 2),
        "galaxy_bools": torch.tensor([1, 1]).reshape(1, 2, 1),
    }
    full_cat = FullCatalog(2, 2, d)

    with pytest.raises(ValueError) as error_info:
        full_cat.to_tile_params(1, 1, ignore_extra_sources=False)
    assert error_info.value.args[0] == "Number of sources per tile exceeds `max_sources_per_tile`."

    # should return only first source in first tile.
    tile_cat = full_cat.to_tile_params(1, 1, ignore_extra_sources=True)
    assert torch.equal(tile_cat.n_sources, torch.tensor([[1, 0], [0, 0]]).reshape(1, 2, 2))
    assert torch.equal(
        tile_cat.locs, torch.tensor([[[0.5, 0.5], [0, 0]], [[0, 0], [0, 0]]]).reshape(1, 2, 2, 1, 2)
    )
    assert torch.equal(
        tile_cat["galaxy_bools"],
        torch.tensor([[[1.0], [0.0]], [[0.0], [0.0]]]).reshape(1, 2, 2, 1, 1),
    )
