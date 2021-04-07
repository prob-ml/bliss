import pytest
import torch
import pytorch_lightning as pl
from hydra.experimental import initialize, compose

from bliss import sleep
from bliss.datasets import simulated, galsim_galaxies
from bliss.models import galaxy_net


# command line arguments for tests
def pytest_addoption(parser):
    parser.addoption("--gpus", default="cpu", type=str, help="--gpus option for trainer.")


def get_cfg(overrides, devices):
    assert "model" in overrides
    overrides = [f"{key}={value}" for key, value in overrides.items()]
    with initialize(config_path="../config"):
        cfg = compose("config", overrides=overrides)
        cfg.training.trainer.update({"gpus": devices.gpus})
    return cfg


class DeviceSetup:
    def __init__(self, gpus):
        self.use_cuda = torch.cuda.is_available() if gpus != "cpu" else False
        self.gpus = gpus if self.use_cuda else None

        # setup device
        self.device = torch.device("cpu")
        if self.gpus and self.use_cuda:
            assert isinstance(self.gpus, str) and len(self.gpus) == 1
            device_id = int(self.gpus[0])
            self.device = torch.device(f"cuda:{device_id}")
            torch.cuda.set_device(self.device)


class SleepSetup:
    def __init__(self, devices):
        self.devices = devices

    def get_cfg(self, overrides):
        return get_cfg(overrides, self.devices)

    def get_dataset(self, overrides):
        cfg = self.get_cfg(overrides)
        return simulated.SimulatedDataset(cfg)

    def get_trainer(self, overrides):
        cfg = self.get_cfg(overrides)
        if cfg.training.deterministic:
            pl.seed_everything(cfg.training.seed)
            return pl.Trainer(**cfg.training.trainer, deterministic=True)
        return pl.Trainer(**cfg.training.trainer)

    def get_sleep(self, overrides):
        cfg = self.get_cfg(overrides)
        return sleep.SleepPhase(cfg)

    def get_trained_sleep(self, overrides):
        cfg = self.get_cfg(overrides)
        dataset = self.get_dataset(overrides)
        trainer = self.get_trainer(overrides)
        sleep_net = sleep.SleepPhase(cfg)
        trainer.fit(sleep_net, datamodule=dataset)
        return sleep_net

    def test_sleep(self, overrides, sleep_net):
        test_module = self.get_dataset(overrides)
        trainer = self.get_trainer(overrides)
        return trainer.test(sleep_net, datamodule=test_module)[0]


class GalaxyVAESetup:
    def __init__(self, devices):
        self.devices = devices

    def get_cfg(self, overrides):
        return get_cfg(overrides, self.devices)

    def get_trained_vae(self, overrides):
        cfg = self.get_cfg(overrides)
        dataset = galsim_galaxies.ToyGaussian(cfg)
        galaxy_vae = galaxy_net.OneCenteredGalaxy(cfg)
        trainer = pl.Trainer(**cfg.training.trainer)
        trainer.fit(galaxy_vae, datamodule=dataset)
        return galaxy_vae.to(self.devices.device)

    def test_vae(self, overrides, galaxy_net):
        cfg = self.get_cfg(overrides)
        test_module = galsim_galaxies.ToyGaussian(cfg)
        trainer = pl.Trainer(**cfg.training.trainer)
        return trainer.test(galaxy_net, datamodule=test_module)[0]


@pytest.fixture(scope="session")
def paths():
    with initialize(config_path="../config"):
        cfg = compose("config")
    return cfg.paths


@pytest.fixture(scope="session")
def devices(pytestconfig):
    gpus = pytestconfig.getoption("gpus")
    return DeviceSetup(gpus)


@pytest.fixture(scope="session")
def sleep_setup(devices):
    return SleepSetup(devices)


@pytest.fixture(scope="session")
def galaxy_vae_setup(devices):
    return GalaxyVAESetup(devices)
