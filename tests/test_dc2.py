import torch
from hydra.utils import instantiate

from bliss.train import train


class TestDC2:
    def test_dc2(self, cfg):
        dataset = instantiate(cfg.surveys.dc2)
        image = dataset.image_id(0)
        assert image.shape[0] == 1

        images = dataset.image_ids()
        assert len(images) == 25

        p_batch = dataset.predict_batch
        assert p_batch["images"].shape[0] == 1

        dc2_obj = dataset.idx(0)
        dc2_tile = dc2_obj["tile_catalog"]
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

        for i in ("images", "background"):
            assert isinstance(dc2_obj[i], torch.Tensor)

    def test_train_on_dc2(self, cfg):
        train_dc2_cfg = cfg.copy()
        train_dc2_cfg.training.data_source = train_dc2_cfg.surveys.dc2
        train_dc2_cfg.training.pretrained_weights = None
        train_dc2_cfg.encoder.bands = [0]
        train(train_dc2_cfg)
