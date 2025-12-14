from pathlib import Path

import pytest
from hydra import compose, initialize


def pytest_collection_modifyitems(items):
    """Reorder tests so run_first tests run before others."""
    run_first_tests = []
    other_tests = []
    for item in items:
        if item.get_closest_marker("run_first"):
            run_first_tests.append(item)
        else:
            other_tests.append(item)
    items[:] = run_first_tests + other_tests


# command line arguments for tests
def pytest_addoption(parser):
    parser.addoption(
        "--gpu",
        action="store_true",
        default=False,
        help="Run tests using gpu.",
    )


@pytest.fixture
def patch_align(monkeypatch):
    """Patch align function with identity to speed up tests."""
    identity = lambda x, *_args, **_kwargs: x
    monkeypatch.setattr("bliss.surveys.survey.align", identity)


@pytest.fixture(scope="function")
def cfg(pytestconfig, tmp_path):
    use_gpu = pytestconfig.getoption("gpu")
    test_data_dir = Path(__file__).parent / "data"

    # pytest-specific overrides
    overrides = {
        "train.trainer.accelerator": "gpu" if use_gpu else "cpu",
        "predict.trainer.accelerator": "gpu" if use_gpu else "cpu",
        "predict.device": "cuda:0" if use_gpu else "cpu",
        "paths.test_data": test_data_dir,
        "paths.output": str(tmp_path / "output"),
        "paths.cached_data": str(tmp_path / "cached_dataset"),
    }
    overrides = [f"{k}={v}" if v is not None else f"{k}=null" for k, v in overrides.items()]
    with initialize(config_path=".", version_base=None):
        the_cfg = compose("testing_config", overrides=overrides)
    return the_cfg
