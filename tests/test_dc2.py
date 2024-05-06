import torch
from einops import rearrange
from hydra.utils import instantiate

from bliss.catalog import TileCatalog
from bliss.encoder.data_augmentation import (
    aug_rotate90,
    aug_rotate180,
    aug_rotate270,
    aug_shift,
    aug_vflip,
)
from bliss.main import train


class TestDC2:
    def test_dc2(self, cfg):
        dataset = instantiate(cfg.surveys.dc2)
        dataset.prepare_data()
        image = dataset.image_id(0)
        dc2_obj = dataset.idx(0)
        dc2_tile = dc2_obj["tile_catalog"]

        assert image.shape[0] == 6

        images = dataset.image_ids()
        assert len(images) == 25

        p_batch = {
            "images": dc2_obj["images"],
            "background": dc2_obj["background"],
        }
        p_batch = dataset.predict_batch
        dataset.predict_batch = p_batch
        assert p_batch["images"].shape[0] == 6

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
            assert isinstance(dc2_tile[k], torch.Tensor)

        for i in ("images", "background", "psf_params"):
            assert isinstance(dc2_obj[i], torch.Tensor)

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
        train(train_dc2_cfg.train)

    def test_dc2_augmentation(self, cfg):
        train_dc2_cfg = cfg.copy()

        dataset = instantiate(train_dc2_cfg.surveys.dc2)
        dataset.prepare_data()
        dc2_obj = dataset.dc2_data[0]

        tile_dict = {}
        dc2_tile = dc2_obj["tile_catalog"]
        tile_dict["locs"] = rearrange(dc2_tile["locs"], "h w nh nw -> 1 h w nh nw")
        tile_dict["n_sources"] = rearrange(dc2_tile["n_sources"], "h w -> 1 h w")
        tile_dict["source_type"] = rearrange(dc2_tile["source_type"], "h w nh nw -> 1 h w nh nw")
        tile_dict["galaxy_fluxes"] = rearrange(
            dc2_tile["galaxy_fluxes"], "h w nh nw -> 1 h w nh nw"
        )
        tile_dict["galaxy_params"] = rearrange(
            dc2_tile["galaxy_params"], "h w nh nw -> 1 h w nh nw"
        )
        tile_dict["star_fluxes"] = rearrange(dc2_tile["star_fluxes"], "h w nh nw -> 1 h w nh nw")
        tile_dict["star_log_fluxes"] = rearrange(
            dc2_tile["star_log_fluxes"], "h w nh nw -> 1 h w nh nw"
        )
        origin_tile = TileCatalog(4, tile_dict)
        origin_full = origin_tile.to_full_catalog()

        imgs = rearrange(dc2_obj["images"], "b h w -> 1 b 1 h w")
        bgs = rearrange(dc2_obj["background"], "b h w -> 1 b 1 h w")

        aug_input_images = [imgs, bgs]
        aug_input_images = torch.cat(aug_input_images, dim=2)

        aug_list = [aug_vflip, aug_rotate90, aug_rotate180, aug_rotate270, aug_shift]

        for aug_method in aug_list:
            aug_image, aug_full = aug_method(origin_full, aug_input_images)
            assert aug_image[0, :, 0, :, :].shape == dc2_obj["images"].shape
            assert aug_image[0, :, 1, :, :].shape == dc2_obj["background"].shape
            assert aug_full["n_sources"] <= origin_full.n_sources

        # test rotatation
        aug_image90, aug_full90 = aug_rotate90(origin_full, aug_input_images)
        _, aug_full270 = aug_rotate270(origin_full, aug_input_images)

        _, aug_full90180 = aug_rotate180(aug_full90, aug_image90)
        _, aug_full90270 = aug_rotate270(aug_full90, aug_image90)

        assert aug_full90270 == origin_full
        assert aug_full90180 == aug_full270
