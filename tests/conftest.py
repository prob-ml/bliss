import pytest
import torch
import pytorch_lightning as pl
from hydra import initialize, compose

from bliss import sleep
from bliss.datasets import simulated, galsim_galaxies
from bliss.models import galaxy_net


# command line arguments for tests
def pytest_addoption(parser):
    parser.addoption(
        "--gpu",
        action="store_true",
        default=False,
        help="Run tests using gpu.",
    )


def pytest_collection_modifyitems(config, items):
    # skip `multi_gpu` required tests when running on cpu or only single gpu is avaiable
    if config.getoption("--gpu") and torch.cuda.device_count() >= 2:
        return
    skip = pytest.mark.skip(reason="need --gpu option and more than 2 available gpus to run")
    for item in items:
        if "multi_gpu" in item.keywords:
            item.add_marker(skip)


def get_cfg(overrides, devices):
    assert "model" in overrides
    overrides = [f"{key}={value}" for key, value in overrides.items()]
    with initialize(config_path="../config"):
        cfg = compose("config", overrides=overrides)
        cfg.training.trainer.update({"gpus": devices.gpus})
    return cfg


class DeviceSetup:
    def __init__(self, use_gpu):
        self.use_cuda = torch.cuda.is_available() if use_gpu else False
        self.gpus = 1 if self.use_cuda else None
        self.device = torch.device("cpu")
        if self.use_cuda:
            self.device = torch.device("cuda:0")
            torch.cuda.set_device(self.device)


class SleepSetup:
    def __init__(self, devices):
        self.devices = devices

    def get_cfg(self, overrides):
        return get_cfg(overrides, self.devices)

    def get_dataset(self, overrides):
        cfg = self.get_cfg(overrides)
        return simulated.SimulatedDataset(**cfg.dataset.kwargs)

    def get_trainer(self, overrides):
        cfg = self.get_cfg(overrides)
        if cfg.training.deterministic:
            pl.seed_everything(cfg.training.seed)
            return pl.Trainer(**cfg.training.trainer, deterministic=True)
        return pl.Trainer(**cfg.training.trainer)

    def get_sleep(self, overrides):
        cfg = self.get_cfg(overrides)
        return sleep.SleepPhase(**cfg.model.kwargs, optimizer_params=cfg.optimizer)

    def get_trained_sleep(self, overrides):
        cfg = self.get_cfg(overrides)
        dataset = self.get_dataset(overrides)
        trainer = self.get_trainer(overrides)
        sleep_net = sleep.SleepPhase(**cfg.model.kwargs, optimizer_params=cfg.optimizer)
        trainer.fit(sleep_net, datamodule=dataset)
        return sleep_net

    def test_sleep(self, overrides, sleep_net):
        test_module = self.get_dataset(overrides)
        trainer = self.get_trainer(overrides)
        return trainer.test(sleep_net, datamodule=test_module)[0]


class GalaxyAESetup:
    def __init__(self, devices):
        self.devices = devices

    def get_cfg(self, overrides):
        return get_cfg(overrides, self.devices)

    @staticmethod
    def get_dataset(cfg):
        if cfg.dataset.name == "ToyGaussian":
            return galsim_galaxies.ToyGaussian(**cfg.dataset.kwargs)
        if cfg.dataset.name == "SDSSGalaxies":
            return galsim_galaxies.SDSSGalaxies(**cfg.dataset.kwargs)

        raise NotImplementedError("Dataset no available")

    def get_trained_ae(self, overrides):
        cfg = self.get_cfg(overrides)

        ds = self.get_dataset(cfg)
        galaxy_ae = galaxy_net.OneCenteredGalaxyAE(
            **cfg.model.kwargs, optimizer_params=cfg.optimizer
        )
        trainer = pl.Trainer(**cfg.training.trainer)
        trainer.fit(galaxy_ae, datamodule=ds)
        return galaxy_ae.to(self.devices.device)

    def test_ae(self, overrides, galaxy_net):
        cfg = self.get_cfg(overrides)
        ds = self.get_dataset(cfg)
        trainer = pl.Trainer(**cfg.training.trainer)
        return trainer.test(galaxy_net, datamodule=ds)[0]


@pytest.fixture(scope="session")
def paths():
    with initialize(config_path="../config"):
        cfg = compose("config")
    return cfg.paths


@pytest.fixture(scope="session")
def devices(pytestconfig):
    use_gpu = pytestconfig.getoption("gpu")
    return DeviceSetup(use_gpu)


@pytest.fixture(scope="session")
def sleep_setup(devices):
    return SleepSetup(devices)


@pytest.fixture(scope="session")
def galaxy_ae_setup(devices):
    return GalaxyAESetup(devices)
