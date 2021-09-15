import pytest
import pytorch_lightning as pl
from hydra.utils import instantiate

from bliss.models.galaxy_encoder import GalaxyEncoder


class TestBasicGalaxyMeasure:
    @pytest.fixture(scope="class")
    def overrides(self, devices):
        return {
            "model": "galaxy_encoder_sdss",
            "dataset": "default" if devices.use_cuda else "cpu",
            "training": "cpu",
            "training.trainer.check_val_every_n_epoch": 1,
            "training.n_epochs": 3,  # plotting coverage.
            "dataset.batch_size": 3,
        }

    def test_simulated(self, devices, overrides, get_config):
        cfg = get_config(overrides, devices)
        galaxy_encoder = GalaxyEncoder(**cfg.model.kwargs)
        dataset = instantiate(cfg.dataset)
        trainer = pl.Trainer(**cfg.training.trainer)
        trainer.fit(galaxy_encoder, datamodule=dataset)
