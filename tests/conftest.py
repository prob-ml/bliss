# pylint: skip-file
import os
import shutil
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

    # pytest-specific overrides
    overrides = {
        "train.trainer.accelerator": "gpu" if use_gpu else "cpu",
        "predict.device": "cuda:0" if use_gpu else "cpu",
        "paths.root": Path(__file__).parents[1].as_posix(),
        "paths.output": str(output_path),
        "cached_simulator.cached_data_path": str(cached_data_path),
        "generate.cached_data_path": str(cached_data_path),
    }
    overrides = [f"{k}={v}" if v is not None else f"{k}=null" for k, v in overrides.items()]
    with initialize(config_path=".", version_base=None):
        the_cfg = compose("testing_config", overrides=overrides)
    return the_cfg


@pytest.fixture(scope="session")
def decals_setup_teardown(cfg):
    # replace if needed
    original_ccds_annotated_path = cfg.paths.decals + "/ccds-annotated-decam-dr9.fits"
    temp_ccds_annotated_path = cfg.paths.decals + "/ccds-annotated-decam-dr9-large.fits"
    large_file_existed = os.path.exists(original_ccds_annotated_path)
    if large_file_existed:
        shutil.move(original_ccds_annotated_path, temp_ccds_annotated_path)
    shutil.copyfile(
        cfg.paths.data + "/decals/ccds-annotated-decam-dr9-small.fits",
        original_ccds_annotated_path,
    )

    yield

    # restore
    os.remove(original_ccds_annotated_path)
    if large_file_existed:
        shutil.move(temp_ccds_annotated_path, original_ccds_annotated_path)


@pytest.fixture(scope="session")
def encoder(cfg):
    encoder = instantiate(cfg.encoder).to(cfg.predict.device)
    enc_state_dict = torch.load(cfg.predict.weight_save_path)
    encoder.load_state_dict(enc_state_dict)
    # remember to put encoder in eval mode if using it for prediction
    return encoder


@pytest.fixture(scope="session")
def decoder(cfg):
    simulator = instantiate(cfg.simulator)
    return simulator.image_decoder


@pytest.fixture(scope="session")
def multiband_dataloader(cfg):
    with open(cfg.paths.data + "/multiband_data/dataset_0.pt", "rb") as f:
        data = torch.load(f)
    return DataLoader(data, batch_size=8, shuffle=False)


@pytest.fixture(scope="session")
def multi_source_dataloader(cfg):
    with open(cfg.paths.data + "/test_multi_source.pt", "rb") as f:
        data = torch.load(f)
    return DataLoader(data, batch_size=8, shuffle=False)
