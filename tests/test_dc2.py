import logging

import torch
from hydra.utils import instantiate

from bliss.catalog import FullCatalog, TileCatalog
from bliss.global_env import GlobalEnv
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

    if not _test_tensor_all_close(left["locs"], right["locs"]):
        logger.warning("locs are different")

    if not _test_tensor_all_close(left["n_sources"], right["n_sources"]):
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
        and _test_tensor_all_close(left["locs"], right["locs"])
        and _test_tensor_all_close(left["n_sources"], right["n_sources"])
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

    if not _test_tensor_all_close(left["plocs"], right["plocs"]):
        logger.warning("plocs are different")

    if not _test_tensor_all_close(left["n_sources"], right["n_sources"]):
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
        and _test_tensor_all_close(left["plocs"], right["plocs"])
        and _test_tensor_all_close(left["n_sources"], right["n_sources"])
        and left.batch_size == right.batch_size
        and left.max_sources == right.max_sources
        and _test_data_equal(left.data, right.data)
    )


class TestDC2:
    def test_dc2_size_and_type(self, cfg):
        dc2 = instantiate(cfg.surveys.dc2)
        dc2.prepare_data()
        dc2.setup(stage="fit")

        # temporarily set global settings
        GlobalEnv.seed_in_this_program = 0
        GlobalEnv.current_encoder_epoch = 0

        dc2 = list(dc2.train_dataloader())

        assert dc2[0]["images"].shape[1] == 6
        assert len(dc2) == 5

        params = (
            "locs",
            "n_sources",
            "source_type",
            "fluxes",
        )

        for k in params:
            assert isinstance(dc2[0]["tile_catalog"][k], torch.Tensor)

        for k in ("images", "psf_params"):
            assert isinstance(dc2[0][k], torch.Tensor)

        # reset global settings to None
        GlobalEnv.seed_in_this_program = None
        GlobalEnv.current_encoder_epoch = None

    def test_train_on_dc2(self, cfg):
        cfg.encoder.survey_bands = ["u", "g", "r", "i", "z", "y"]
        cfg.train.data_source = cfg.surveys.dc2
        cfg.train.pretrained_weights = None
        cfg.encoder.image_normalizers.psf.num_psf_params = 4

        cfg.encoder.var_dist.factors = [
            {
                "_target_": "bliss.encoder.variational_dist.BernoulliFactor",
                "name": "n_sources",
                "sample_rearrange": None,
                "nll_rearrange": None,
                "nll_gating": None,
            },
            {
                "_target_": "bliss.encoder.variational_dist.TDBNFactor",
                "name": "locs",
                "sample_rearrange": "b ht wt d -> b ht wt 1 d",
                "nll_rearrange": "b ht wt 1 d -> b ht wt d",
                "nll_gating": {"_target_": "bliss.encoder.variational_dist.SourcesGating"},
            },
            {
                "_target_": "bliss.encoder.variational_dist.BernoulliFactor",
                "name": "source_type",
                "sample_rearrange": "b ht wt -> b ht wt 1 1",
                "nll_rearrange": "b ht wt 1 1 -> b ht wt",
                "nll_gating": {"_target_": "bliss.encoder.variational_dist.SourcesGating"},
            },
            {
                "_target_": "bliss.encoder.variational_dist.LogNormalFactor",
                "name": "fluxes",
                "dim": 6,
                "sample_rearrange": "b ht wt d -> b ht wt 1 d",
                "nll_rearrange": "b ht wt 1 d -> b ht wt d",
                "nll_gating": {"_target_": "bliss.encoder.variational_dist.SourcesGating"},
            },
        ]

        for f in cfg.variational_factors:
            if f.name == "fluxes":
                f.dim = 6

        train(cfg.train)
