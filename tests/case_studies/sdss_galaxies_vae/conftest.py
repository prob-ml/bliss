from pathlib import Path

import pytest
from hydra import compose, initialize

from tests.conftest import ModelSetup


def get_vae_cfg(overrides):
    overrides.update({"paths.root": Path(__file__).parents[3].as_posix()})
    overrides = [f"{k}={v}" if v is not None else f"{k}=null" for k, v in overrides.items()]
    config_path = "../../../case_studies/sdss_galaxies_vae/config"
    with initialize(config_path=config_path, version_base=None):
        cfg = compose("config", overrides=overrides)
    return cfg


class VAESetup(ModelSetup):
    def get_cfg(self, overrides):
        return get_vae_cfg(overrides)


@pytest.fixture(scope="session")
def vae_setup(devices):
    return VAESetup(devices)
