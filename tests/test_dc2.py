import logging

import torch
from einops import rearrange
from hydra.utils import instantiate

from bliss.catalog import FullCatalog, TileCatalog
from bliss.encoder.data_augmentation import (
    aug_rotate90,
    aug_rotate180,
    aug_rotate270,
    aug_shift,
    aug_vflip,
)
from bliss.main import train


def _test_tensor_all_close(left, right):
    return torch.allclose(left, right)


def _test_data_equal(left_data, right_data):
    logger = logging.getLogger(__name__)
    is_equal = True
    for k, v in left_data.items():
        cur_test_equal = _test_tensor_all_close(right_data[k], v)
        if not cur_test_equal:
            logger.warning("%s are different", k)
        is_equal &= cur_test_equal
    return is_equal


def _test_tile_catalog_equal(left, right):
    assert isinstance(left, TileCatalog), "left is not TileCatalog"
    assert isinstance(right, TileCatalog), "right is not TileCatalog"

    logger = logging.getLogger(__name__)

    if left.tile_slen != right.tile_slen:
        logger.warning(
            "tile_slen are different:   left: %d; right: %d",
            left.tile_slen,
            right.tile_slen,
        )

    if not _test_tensor_all_close(left.locs, right.locs):
        logger.warning("locs are different")

    if not _test_tensor_all_close(left.n_sources, right.n_sources):
        logger.warning("n_sources are different")

    if left.batch_size != right.batch_size:
        logger.warning(
            "batch_size are different:  left: %d; right: %d",
            left.batch_size,
            right.batch_size,
        )

    if left.n_tiles_h != right.n_tiles_h:
        logger.warning(
            "n_tiles_h are different:   left: %d; right: %d",
            left.n_tiles_h,
            right.n_tiles_h,
        )

    if right.n_tiles_w != right.n_tiles_w:
        logger.warning(
            "n_tiles_w are different:   left: %d; right: %d",
            left.n_tiles_w,
            right.n_tiles_w,
        )

    if left.max_sources != right.max_sources:
        logger.warning(
            "max_sources are different: left: %d; right: %d",
            left.max_sources,
            right.max_sources,
        )

    _test_data_equal(left.data, right.data)

    return (
        left.tile_slen == right.tile_slen  # noqa: WPS222
        and _test_tensor_all_close(left.locs, right.locs)
        and _test_tensor_all_close(left.n_sources, right.n_sources)
        and left.batch_size == right.batch_size
        and left.n_tiles_h == right.n_tiles_h
        and left.n_tiles_w == right.n_tiles_w
        and left.max_sources == right.max_sources
        and _test_data_equal(left.data, right.data)
    )


def _test_full_catalog_equal(left, right):
    assert isinstance(left, FullCatalog), "left is not FullCatalog"
    assert isinstance(right, FullCatalog), "right is not FullCatalog"

    logger = logging.getLogger(__name__)

    if left.height != right.height:
        logger.warning(
            "heights are different: left: %d; right: %d",
            left.height,
            right.height,
        )

    if left.width != right.width:
        logger.warning(
            "widths are different:  left: %d; right: %d",
            left.width,
            right.width,
        )

    if not _test_tensor_all_close(left.plocs, right.plocs):
        logger.warning("plocs are different")

    if not _test_tensor_all_close(left.n_sources, right.n_sources):
        logger.warning("n_sources are different")

    if left.batch_size != right.batch_size:
        logger.warning(
            "batch_size are different:  left: %d; right: %d",
            left.batch_size,
            right.batch_size,
        )

    if left.max_sources != right.max_sources:
        logger.warning(
            "max_sources are different: left: %d; right: %d",
            left.max_sources,
            right.max_sources,
        )

    _test_data_equal(left.data, right.data)

    return (
        left.height == right.height  # noqa:WPS222
        and left.width == right.width
        and _test_tensor_all_close(left.plocs, right.plocs)
        and _test_tensor_all_close(left.n_sources, right.n_sources)
        and left.batch_size == right.batch_size
        and left.max_sources == right.max_sources
        and _test_data_equal(left.data, right.data)
    )


class TestDC2:
    def test_dc2_size_and_type(self, cfg):
        dc2 = instantiate(cfg.surveys.dc2)
        dc2.prepare_data()
        dc2.setup()

        assert dc2[0]["images"].shape[0] == 6
        assert len(dc2.image_ids()) == 25

        params = (
            "locs",
            "n_sources",
            "source_type",
            "galaxy_fluxes",
            "galaxy_params",
            "star_fluxes",
            "star_log_fluxes",
        )

        for k in params:
            assert isinstance(dc2[0]["tile_catalog"][k], torch.Tensor)

        for k in ("images", "background", "psf_params"):
            assert isinstance(dc2[0][k], torch.Tensor)

    def test_train_on_dc2(self, cfg):
        train_dc2_cfg = cfg.copy()
        train_dc2_cfg.encoder.image_normalizer.bands = [0, 1, 2, 3, 4, 5]
        # why are these bands out of order? (should be "ugrizy") why does the test break if they
        # are ordered correctly?
        train_dc2_cfg.encoder.survey_bands = ["g", "i", "r", "u", "y", "z"]
        train_dc2_cfg.train.data_source = train_dc2_cfg.surveys.dc2
        train_dc2_cfg.encoder.do_data_augmentation = True
        train_dc2_cfg.train.pretrained_weights = None
        # log transform doesn't work in this test because the DC2 background is sometimes negative.
        # why would the background be negative? are we using the wrong background estimate?
        train_dc2_cfg.encoder.image_normalizer.log_transform_stdevs = []
        train_dc2_cfg.encoder.image_normalizer.asinh_params = {
            "scale": 0.1,
            "thresholds": [-3, 0, 1, 3],
        }
        train(train_dc2_cfg.train)

    def test_dc2_augmentation(self, cfg):
        dc2 = instantiate(cfg.surveys.dc2)
        dc2.prepare_data()
        dc2.setup()

        dc2_first_data = dc2[0]
        tile_dict = dc2_first_data["tile_catalog"]

        for k, v in tile_dict.items():
            if k != "n_sources":
                tile_dict[k] = rearrange(v, "h w nh nw -> 1 h w nh nw")
        tile_dict["n_sources"] = rearrange(tile_dict["n_sources"], "h w -> 1 h w")

        ori_tile = TileCatalog(4, tile_dict)
        ori_full = ori_tile.to_full_catalog()

        imgs = rearrange(dc2_first_data["images"], "b h w -> 1 b 1 h w")
        bgs = rearrange(dc2_first_data["background"], "b h w -> 1 b 1 h w")

        aug_input_images = [imgs, bgs]
        aug_input_images = torch.cat(aug_input_images, dim=2)

        aug_list = [aug_vflip, aug_rotate90, aug_rotate180, aug_rotate270, aug_shift]

        for aug_method in aug_list:
            aug_image, aug_full = aug_method(ori_full, aug_input_images)
            assert aug_image[0, :, 0, :, :].shape == dc2_first_data["images"].shape
            assert aug_image[0, :, 1, :, :].shape == dc2_first_data["background"].shape
            assert aug_full["n_sources"] <= ori_full.n_sources

        # test rotatation
        aug_image90, aug_full90 = aug_rotate90(ori_full, aug_input_images)
        _, aug_full270 = aug_rotate270(ori_full, aug_input_images)

        _, aug_full90180 = aug_rotate180(aug_full90, aug_image90)
        _, aug_full90270 = aug_rotate270(aug_full90, aug_image90)

        assert _test_full_catalog_equal(aug_full90270, ori_full)
        assert _test_full_catalog_equal(aug_full90180, aug_full270)
