from pathlib import Path

import pytest
import pytorch_lightning as pl
import torch
from hydra import compose, initialize
from hydra.utils import instantiate


# command line arguments for tests
def pytest_addoption(parser):
    parser.addoption(
        "--gpu",
        action="store_true",
        default=False,
        help="Run tests using gpu.",
    )


def get_cfg(overrides, devices):
    overrides.update({"gpus": devices.gpus})
    overrides.update(
        {"training.weight_save_path": None, "paths.root": Path(__file__).parents[1].as_posix()}
    )
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
        overrides.update(
            {
                # Testing-specific overrides
                "+training.seed": 42,
                "training.trainer.logger": False,
                "training.trainer.check_val_every_n_epoch": 1001,
                "training.trainer.deterministic": not self.devices.use_cuda,
            }
        )
        cfg = self.get_cfg(overrides)
        if cfg.training.trainer.deterministic:
            pl.seed_everything(cfg.training.seed)
        return instantiate(cfg.training.trainer)

    def get_dataset_for_training(self, overrides):
        cfg = self.get_cfg(overrides)
        return instantiate(cfg.training.dataset)

    def get_model_for_training(self, overrides):
        cfg = self.get_cfg(overrides)
        model = instantiate(cfg.training.model, optimizer_params=cfg.training.optimizer_params)
        return model.to(self.devices.device)

    def get_trained_model(self, overrides):
        dataset = self.get_dataset_for_training(overrides)
        trainer = self.get_trainer(overrides)
        model = self.get_model_for_training(overrides)
        trainer.fit(model, datamodule=dataset)
        return model.to(self.devices.device)

    def test_model(self, overrides, model):
        test_module = self.get_dataset_for_training(overrides)
        trainer = self.get_trainer(overrides)
        return trainer.test(model, datamodule=test_module)[0]


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
