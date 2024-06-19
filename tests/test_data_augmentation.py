from copy import deepcopy

import torch

from bliss.cached_dataset import FullCatalogToTileTransform
from bliss.data_augmentation import RandomShiftTransform, RotateFlipTransform


def test_rotate_flip(cfg):
    datum_lst = torch.load(cfg.paths.test_data + "/two_image_cached_dataset.pt")

    fcttt = FullCatalogToTileTransform(2, 6)
    datum = fcttt(datum_lst[0])

    original_datum = deepcopy(datum)
    fr_transform = RotateFlipTransform()

    # rotate/flip at random (smoke test)
    fr_transform(datum)

    # rotate 90 degrees, 4 times
    for _ in range(4):
        datum = fr_transform(datum, rotate_id=1, do_flip=False)
    assert original_datum["images"].allclose(datum["images"])
    assert original_datum["tile_catalog"]["n_sources"].allclose(datum["tile_catalog"]["n_sources"])
    assert original_datum["tile_catalog"]["locs"].allclose(datum["tile_catalog"]["locs"])

    # rotate 180 degrees twice
    datum = fr_transform(datum, rotate_id=2, do_flip=False)
    assert not original_datum["tile_catalog"]["locs"].allclose(datum["tile_catalog"]["locs"])
    datum = fr_transform(datum, rotate_id=2, do_flip=False)
    assert original_datum["images"].allclose(datum["images"])
    assert original_datum["background"].allclose(datum["background"])
    assert original_datum["tile_catalog"]["locs"].allclose(datum["tile_catalog"]["locs"])

    # rotate 270 degrees and flip, then flip, then rotate 90 degrees
    datum = fr_transform(datum, rotate_id=3, do_flip=True)
    assert not original_datum["tile_catalog"]["locs"].allclose(datum["tile_catalog"]["locs"])
    datum = fr_transform(datum, rotate_id=0, do_flip=True)
    datum = fr_transform(datum, rotate_id=1, do_flip=False)
    assert original_datum["images"].allclose(datum["images"])
    assert original_datum["tile_catalog"]["locs"].allclose(datum["tile_catalog"]["locs"])


def test_rotate_with_toy_data(cfg):
    # create an 10x10 image, with 2x2 tiles
    d = {
        "locs": torch.zeros(5, 5, 1, 2),
        "n_sources": torch.zeros(5, 5),
    }
    d["n_sources"][0, 0] = 1
    d["locs"][0, 0, 0, 0] = 0.1
    d["locs"][0, 0, 0, 1] = 0.1
    datum = {
        "images": torch.zeros(1, 10, 10),
        "background": torch.zeros(1, 10, 10),
        "tile_catalog": d,
        "psf_params": None,
    }
    datum["images"][0, 0, 0] = 42
    fr_transform = RotateFlipTransform()
    rotated_datum = fr_transform(datum, rotate_id=1, do_flip=False)

    assert rotated_datum["images"][0, 9, 0].isclose(torch.tensor([42.0]))
    assert rotated_datum["tile_catalog"]["n_sources"][4, 0] == 1
    assert rotated_datum["tile_catalog"]["locs"][4, 0, 0].allclose(torch.tensor([0.9, 0.1]))


def test_random_shift(cfg):
    datum_lst = torch.load(cfg.paths.test_data + "/two_image_cached_dataset.pt")

    fcttt = FullCatalogToTileTransform(2, 6)
    datum = fcttt(datum_lst[0])

    original_datum = deepcopy(datum)
    rs_transform = RandomShiftTransform(2, 6)

    # shift at random (smoke test)
    rs_transform(datum)

    # shift up and left, then down and right
    datum = rs_transform(datum, vertical_shift=1, horizontal_shift=-1)
    datum = rs_transform(datum, vertical_shift=-1, horizontal_shift=1)
    assert torch.allclose(original_datum["images"][2:110, 2:110], datum["images"][2:110, 2:110])
    # the order of sources can change in tiles with multiple sources
    assert torch.allclose(
        original_datum["tile_catalog"]["locs"][1:55, 1:55].sort(2).values,
        datum["tile_catalog"]["locs"][1:55, 1:55].sort(2).values,
        atol=1e-5,
    )
