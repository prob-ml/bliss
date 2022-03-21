import pytest

from bliss.train import train


class TestBasicGalaxyMeasure:
    @pytest.fixture(scope="class")
    def overrides(self, devices):
        overrides = {
            "mode": "train",
            "training": "sdss_galaxy_encoder",
            "training.trainer.check_val_every_n_epoch": 1,
            "training.n_epochs": 3,  # plotting coverage.
            "datasets.simulated.batch_size": 3,
        }
        if not devices.use_cuda:
            overrides.update(
                {
                    "datasets.simulated.n_batches": 1,
                    "datasets.simulated.generate_device": "cpu",
                    "datasets.simulated.testing_file": None,
                }
            )
        return overrides

    def test_simulated(self, devices, overrides, get_sdss_galaxies_config):
        cfg = get_sdss_galaxies_config(overrides, devices)
        train(cfg)
