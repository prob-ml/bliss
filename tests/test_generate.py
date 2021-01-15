import pytest
from bliss import generate
from hydra.experimental import initialize, compose


def test_generate_run():
    with initialize(config_path="../config"):
        cfg = compose("config")
        generate.main(cfg)
