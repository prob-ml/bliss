from pathlib import Path

import pytest
from hydra import compose, initialize

from tests.conftest import ModelSetup


def get_vae_cfg(overrides, devices):
    overrides.update({"gpus": devices.gpus, "paths.root": Path(__file__).parents[3].as_posix()})
    overrides = [f"{k}={v}" if v is not None else f"{k}=null" for k, v in overrides.items()]
    with initialize(config_path="../../../case_studies/sdss_galaxies_vae/config"):
        cfg = compose("config", overrides=overrides)
    return cfg


class VAESetup(ModelSetup):
    def get_cfg(self, overrides):
        return get_vae_cfg(overrides, self.devices)


@pytest.fixture(scope="session")
def vae_setup(devices):
    return VAESetup(devices)
