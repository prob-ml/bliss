# pylint: skip-file
from pathlib import Path

import pytest
from hydra import compose, initialize


# command line arguments for tests
def pytest_addoption(parser):
    parser.addoption(
        "--gpu",
        action="store_true",
        default=False,
        help="Run tests using gpu.",
    )


# TODO: load trained encoder here and make it available as a fixture


@pytest.fixture(scope="session")
def cfg(pytestconfig):
    use_gpu = pytestconfig.getoption("gpu")

    # pytest-specific overrides
    overrides = {
        "training.seed": 42,
        "training.trainer.logger": False,
        "training.trainer.check_val_every_n_epoch": 1001,
        "training.trainer.enable_checkpointing": False,
        "training.weight_save_path": None,
        "training.trainer.profiler": None,
        "training.trainer.accelerator": "gpu" if use_gpu else "cpu",
        "paths.root": Path(__file__).parents[2].as_posix(),
    }
    overrides = [f"{k}={v}" if v is not None else f"{k}=null" for k, v in overrides.items()]
    with initialize(config_path=".", version_base=None):
        the_cfg = compose("star_basic", overrides=overrides)
    return the_cfg
