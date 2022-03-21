import pytest

from bliss.train import train


class TestSdssBlendedGalaxies:
    @pytest.fixture(scope="class")
    def overrides(self, devices):
        return {
            "mode": "train",
            "training": "sdss_galaxy_encoder_real",
            "datasets.sdss_blended_galaxies.prerender_device": "cuda"
            if devices.use_cuda
            else "cpu",
            "+datasets.sdss_blended_galaxies.slen": 40,
            "+datasets.sdss_blended_galaxies.h_start": 200,
            "+datasets.sdss_blended_galaxies.w_start": 1800,
            "+datasets.sdss_blended_galaxies.scene_size": 100,
            "training.n_epochs": 1,
            "training.trainer.log_every_n_steps": 1,
        }

    def test_sdss_blended_galaxies(self, devices, overrides, get_sdss_galaxies_config):
        cfg = get_sdss_galaxies_config(overrides, devices)
        train(cfg)
