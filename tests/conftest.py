# pylint: skip-file
from pathlib import Path

import pytest
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


@pytest.fixture(scope="session")
def cfg(pytestconfig):
    use_gpu = pytestconfig.getoption("gpu")

    # pytest-specific overrides
    overrides = {
        "training.trainer.accelerator": "gpu" if use_gpu else "cpu",
        "predict.device": "cuda:0" if use_gpu else "cpu",
        "paths.root": Path(__file__).parents[1].as_posix(),
    }
    overrides = [f"{k}={v}" if v is not None else f"{k}=null" for k, v in overrides.items()]
    with initialize(config_path=".", version_base=None):
        the_cfg = compose("testing_config", overrides=overrides)
    return the_cfg


@pytest.fixture(scope="session")
def encoder(cfg):
    encoder = instantiate(cfg.encoder).to(cfg.predict.device)
    enc_state_dict = torch.load(cfg.predict.weight_save_path)
    encoder.load_state_dict(enc_state_dict)
    # remember to put encoder in eval mode if using it for prediction
    return encoder


@pytest.fixture(scope="session")
def decoder(cfg):
    return instantiate(cfg.simulator.decoder)
