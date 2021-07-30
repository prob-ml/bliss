import pytest
import torch
import pytorch_lightning as pl
from hydra import initialize, compose

from bliss import sleep
from bliss.datasets import simulated, galsim_galaxies
from bliss.models import galaxy_net, galaxy_encoder, flux_net


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
    overrides.update({"gpus": devices.gpus})
    overrides = [f"{k}={v}" if v is not None else f"{k}=null" for k, v in overrides.items()]
    with initialize(config_path="../config"):
        cfg = compose("config", overrides=overrides)
    return cfg


class DeviceSetup:
    def __init__(self, use_gpu):
        self.use_cuda = torch.cuda.is_available() if use_gpu else False
        self.gpus = 1 if self.use_cuda else None
        self.device = torch.device("cpu")
        if self.use_cuda:
            self.device = torch.device("cuda:0")
            torch.cuda.set_device(self.device)


class ModelSetup:
    def __init__(self, devices):
        self.devices = devices

    def get_cfg(self, overrides):
        return get_cfg(overrides, self.devices)

    def get_trainer(self, overrides):
        cfg = self.get_cfg(overrides)
        if cfg.training.deterministic:
            pl.seed_everything(cfg.training.seed)
            return pl.Trainer(**cfg.training.trainer, deterministic=True)
        return pl.Trainer(**cfg.training.trainer)

    def get_dataset(self, overrides):
        cfg = self.get_cfg(overrides)
        ds_name = cfg.dataset.name
        if ds_name == "ToyGaussian":
            return galsim_galaxies.ToyGaussian(**cfg.dataset.kwargs)
        if ds_name == "SDSSGalaxies":
            return galsim_galaxies.SDSSGalaxies(**cfg.dataset.kwargs)
        if ds_name == "SimulatedDataset":
            return simulated.SimulatedDataset(**cfg.dataset.kwargs)
        raise NotImplementedError()

    def get_model(self, overrides):
        cfg = self.get_cfg(overrides)
        opt = cfg.optimizer
        model_name = cfg.model.name

        if model_name == "SleepPhase":
            model = sleep.SleepPhase(**cfg.model.kwargs, optimizer_params=opt)
        elif model_name == "GalaxyEncoder":
            model = galaxy_encoder.GalaxyEncoder(**cfg.model.kwargs, optimizer_params=opt)
        elif model_name == "FluxEncoder":
            model = flux_net.FluxEstimator(**cfg.model.kwargs, optimizer_params=opt)
        elif model_name == "OneCenteredGalaxyAE":
            model = galaxy_net.OneCenteredGalaxyAE(**cfg.model.kwargs, optimizer_params=opt)
        else:
            raise NotImplementedError()

        return model.to(self.devices.device)

    def get_trained_model(self, overrides):
        dataset = self.get_dataset(overrides)
        trainer = self.get_trainer(overrides)
        model = self.get_model(overrides)
        trainer.fit(model, datamodule=dataset)
        return model.to(self.devices.device)

    def test_model(self, overrides, model):
        test_module = self.get_dataset(overrides)
        trainer = self.get_trainer(overrides)
        return trainer.test(model, datamodule=test_module)[0]


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
def model_setup(devices):
    return ModelSetup(devices)


@pytest.fixture(scope="session")
def get_config():
    return get_cfg
