from pathlib import Path

import pytest
from hydra import compose, initialize


def get_strong_lensing_cfg(overrides):
    overrides.update({"paths.root": Path(__file__).parents[3].as_posix()})
    overrides = [f"{k}={v}" if v is not None else f"{k}=null" for k, v in overrides.items()]
    with initialize(config_path="../../../case_studies/strong_lensing/config", version_base=None):
        cfg = compose("config", overrides=overrides)
    return cfg


class SingleGalsimGalaxiesSetup:
    def get_cfg(self, overrides):
        return get_strong_lensing_cfg(overrides)


@pytest.fixture(scope="session")
def strong_lensing_setup(devices):
    return SingleGalsimGalaxiesSetup(devices)


@pytest.fixture(scope="session")
def get_strong_lensing_config():
    return get_strong_lensing_cfg