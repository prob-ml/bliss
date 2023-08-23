import torch
from einops import rearrange
from hydra.utils import instantiate

from bliss import data_augmentation
from bliss.catalog import TileCatalog
from bliss.train import train


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
        train_dc2_cfg.encoder.bands = [0, 1, 2, 3, 4, 5]
        train_dc2_cfg.encoder.survey_bands = ["g", "i", "r", "u", "y", "z"]
        train_dc2_cfg.training.data_source = train_dc2_cfg.surveys.dc2
        train_dc2_cfg.encoder.input_transform_params.use_deconv_channel = True
        train_dc2_cfg.encoder.data_augmentation.epoch_start = 0
        train_dc2_cfg.training.pretrained_weights = None
        train(train_dc2_cfg)

    def test_dc2_augmentation(self, cfg):
        train_dc2_cfg = cfg.copy()
        train_dc2_cfg.encoder.input_transform_params.use_deconv_channel = True

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
        origin_full = origin_tile.to_full_params()

        image = rearrange(dc2_obj["images"], "b h w -> 1 b h w")
        deconv_image = rearrange(dc2_obj["deconvolution"], "b h w -> 1 b h w")

        aug_list = [
            data_augmentation.aug_vflip,
            data_augmentation.aug_hflip,
            data_augmentation.aug_rotate90,
            data_augmentation.aug_rotate180,
            data_augmentation.aug_rotate270,
            data_augmentation.aug_shift,
        ]

        for i in aug_list:
            aug_image, aug_tile, aug_deconv = i(origin_full, image, deconv_image)
            assert aug_image.shape == image.shape
            assert aug_deconv.shape == deconv_image.shape
            assert aug_tile["n_sources"].sum() <= origin_full.n_sources
