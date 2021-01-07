import pytest
import pathlib
import torch
import pytorch_lightning as pl
from hydra.experimental import initialize, compose

import bliss
from bliss import sleep
from bliss.datasets import simulated, catsim
from bliss.models import galaxy_net


# command line arguments for tests
def pytest_addoption(parser):
    parser.addoption("--gpus", default="0", type=str, help="--gpus option for trainer.")
    parser.addoption(
        "--repeat", default=1, type=str, help="Number of times to repeat each test"
    )
    parser.addoption(
        "--runslow", action="store_true", default=False, help="run slow tests"
    )


# add slow marker.
def pytest_configure(config):
    config.addinivalue_line("markers", "slow: mark test as slow to run")


# make --runslow option work with the marker.
def pytest_collection_modifyitems(config, items):
    if config.getoption("--runslow"):
        # --runslow given in cli: do not skip slow tests
        return
    skip_slow = pytest.mark.skip(reason="need --runslow option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)


# allows test repetition with --repeat flag.
def pytest_generate_tests(metafunc):
    if metafunc.config.option.repeat is not None:
        count = int(metafunc.config.option.repeat)

        # We're going to duplicate these tests by parametrizing them,
        # which requires that each test has a fixture to accept the parameter.
        # We can add a new fixture like so:
        metafunc.fixturenames.append("tmp_ct")

        # Now we parametrize. This is what happens when we do e.g.,
        # @pytest.mark.parametrize('tmp_ct', range(count))
        # def test_foo(): pass
        metafunc.parametrize("tmp_ct", range(count))


class DeviceSetup:
    def __init__(self, gpus):
        self.use_cuda = torch.cuda.is_available()
        self.gpus = gpus if self.use_cuda else None

        # setup device
        self.device = torch.device("cpu")
        if self.gpus and self.use_cuda:
            assert isinstance(self.gpus, str) and len(self.gpus) == 1
            device_id = int(self.gpus[0])
            self.device = torch.device(f"cuda:{device_id}")
            torch.cuda.set_device(self.device)


def get_cfg(overrides, devices):
    assert "model" in overrides
    overrides = [f"{key}={value}" for key, value in overrides.items()]
    with initialize(config_path="../config"):
        cfg = compose("config", overrides=overrides)
        cfg.training.trainer.update({"gpus": devices.gpus})
    return cfg


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


class GalaxySetup:
    def __init__(self, devices):
        self.devices = devices

    def get_cfg(self, overrides):
        return get_cfg(overrides, self.devices)

    def get_trained_vae(self, overrides):
        cfg = self.get_cfg(overrides)
        dataset = catsim.SavedCatsim(cfg)
        galaxy_vae = galaxy_net.OneCenteredGalaxy(cfg)
        trainer = pl.Trainer(**cfg.training.trainer)
        trainer.fit(galaxy_vae, datamodule=dataset)
        return galaxy_vae.to(self.devices.device)


# available fixtures provided globally for all tests.
@pytest.fixture(scope="session")
def paths():
    root_path = pathlib.Path(__file__).parent.parent.absolute()
    return {
        "data": root_path.joinpath("data"),
        "model_dir": root_path.joinpath("trials_result"),
    }


@pytest.fixture(scope="session")
def devices(pytestconfig):
    gpus = pytestconfig.getoption("gpus")
    return DeviceSetup(gpus)


@pytest.fixture(scope="session")
def sleep_setup(devices):
    return SleepSetup(devices)


@pytest.fixture(scope="session")
def galaxy_setup(devices):
    return GalaxySetup(devices)
