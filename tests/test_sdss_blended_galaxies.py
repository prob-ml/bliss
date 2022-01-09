import pytest
from hydra.utils import instantiate


class TestSdssBlendedGalaxies:
    @pytest.fixture(scope="class")
    def overrides(self, devices):
        return {
            "+experiment": "sdss_galaxy_encoder_real",
            "dataset.prerender_device": "cuda" if devices.use_cuda else "cpu",
            "dataset.cache_path": None,
            "+dataset.slen": 40,
            "+dataset.h_start": 200,
            "+dataset.w_start": 1800,
            "+dataset.scene_size": 200,
            "training.n_epochs": 1,
        }

    def test_sdss_blended_galaxies(self, devices, overrides, get_config):
        cfg = get_config(overrides, devices)
        galaxy_encoder = instantiate(cfg.model)
        dataset = instantiate(cfg.dataset)
        trainer = instantiate(cfg.training.trainer)
        trainer.fit(galaxy_encoder, datamodule=dataset)
