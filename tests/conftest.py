import pytest
import pathlib
import torch
import pytorch_lightning as pl
from hydra.experimental import initialize, compose

from bliss import sleep
from bliss.datasets import simulated


# command line arguments for tests
def pytest_addoption(parser):

    parser.addoption(
        "--gpus", default="0,", type=str, help="--gpus option for trainer."
    )

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
            device_id = self.gpus.split(",")
            assert len(device_id) == 2 and device_id[1] == ""
            device_id = int(self.gpus[0])
            self.device = torch.device(f"cuda:{device_id}")
            torch.cuda.set_device(self.device)


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
def get_dataset():
    def _dataset(overrides: dict):
        assert "model" in overrides
        overrides = [f"{key}={value}" for key, value in overrides.items()]
        with initialize(config_path="../config"):
            cfg = compose("config", overrides=overrides)
            dataset = simulated.SimulatedDataset(cfg)
        return dataset

    return _dataset


@pytest.fixture(scope="session")
def train_sleep(devices):
    def _encoder(overrides: dict):
        assert "model" in overrides
        overrides = [f"{key}={value}" for key, value in overrides.items()]
        with initialize(config_path="../config"):
            cfg = compose("config", overrides=overrides)
            cfg.training.trainer.update({"gpus": devices.gpus})
            datamodule = simulated.SimulatedModule(cfg)
            sleep_net = sleep.SleepPhase(cfg)
            trainer = pl.Trainer(**cfg.training.trainer)

            # train and then test
            trainer.fit(sleep_net, datamodule=datamodule)
            results = trainer.test(sleep_net, datamodule=datamodule)
            return sleep_net, results

    return _encoder
