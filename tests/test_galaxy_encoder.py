import pytest
import pytorch_lightning as pl
from bliss.models.galaxy_encoder import GalaxyEncoder
from bliss.datasets.simulated import SimulatedDataset


class TestBasicGalaxyMeasure:
    @pytest.fixture(scope="class")
    def overrides(self, devices):
        overrides = {
            "model": "galenc_sdss",
            "dataset": "default" if devices.use_cuda else "cpu",
            "training": "cpu",
            "training.trainer.check_val_every_n_epoch": 1,
            "training.n_epochs": 3,  # plotting coverage.
            "dataset.kwargs.batch_size": 10,
        }
        return overrides

    def test_simulated(self, devices, overrides, get_config):
        cfg = get_config(overrides, devices)
        galaxy_encoder = GalaxyEncoder(**cfg.model.kwargs, optimizer_params=cfg.optimizer)
        dataset = SimulatedDataset(**cfg.dataset.kwargs)
        trainer = pl.Trainer(**cfg.training.trainer)
        trainer.fit(galaxy_encoder, datamodule=dataset)
