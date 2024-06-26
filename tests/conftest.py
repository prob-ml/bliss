from pathlib import Path

import pytest
import torch
from hydra import compose, initialize
from hydra.utils import instantiate
from torch.utils.data import DataLoader


# command line arguments for tests
def pytest_addoption(parser):
    parser.addoption(
        "--gpu",
        action="store_true",
        default=False,
        help="Run tests using gpu.",
    )


@pytest.fixture(scope="session")
def cached_data_path(tmpdir_factory):
    return tmpdir_factory.mktemp("cached_dataset")


@pytest.fixture(scope="session")
def output_path(tmpdir_factory):
    return tmpdir_factory.mktemp("output")


@pytest.fixture(scope="session")
def cfg(pytestconfig, cached_data_path, output_path):
    use_gpu = pytestconfig.getoption("gpu")
    test_data_dir = Path(__file__).parent / "data"

    # pytest-specific overrides
    overrides = {
        "train.trainer.accelerator": "gpu" if use_gpu else "cpu",
        "predict.trainer.accelerator": "gpu" if use_gpu else "cpu",
        "predict.device": "cuda:0" if use_gpu else "cpu",
        "paths.test_data": test_data_dir,
        "paths.output": str(output_path),
        "paths.cached_data": str(cached_data_path),
        "generate.cached_data_path": str(cached_data_path),
    }
    overrides = [f"{k}={v}" if v is not None else f"{k}=null" for k, v in overrides.items()]
    with initialize(config_path=".", version_base=None):
        the_cfg = compose("testing_config", overrides=overrides)
    return the_cfg


@pytest.fixture(scope="session")
def encoder(cfg):
    encoder = instantiate(cfg.encoder).to(cfg.predict.device)
    enc_state_dict = torch.load(cfg.predict.weight_save_path, map_location=cfg.predict.device)
    encoder.load_state_dict(enc_state_dict)
    # remember to put encoder in eval mode if using it for prediction
    return encoder


@pytest.fixture(scope="session")
def decoder(cfg):
    simulator = instantiate(cfg.simulator)
    return simulator.image_decoder


@pytest.fixture(scope="session")
def multiband_dataloader(cfg):
    with open(cfg.paths.test_data + "/multiband_data/dataset_0.pt", "rb") as f:
        data = torch.load(f)
    return DataLoader(data, batch_size=8, shuffle=False)


@pytest.fixture(scope="session")
def multi_source_dataloader(cfg):
    with open(cfg.paths.test_data + "/test_multi_source.pt", "rb") as f:
        data = torch.load(f)
    return DataLoader(data, batch_size=8, shuffle=False)
