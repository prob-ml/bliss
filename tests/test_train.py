from bliss import train
import pytest
from hydra.experimental import initialize, compose


class TestTrain:
    def test_train_run(self, devices):
        overrides = {"training": "default" if devices.use_cuda else "cpu"}
        overrides = [f"{k}={v}" for k, v in overrides.items()]
        with initialize(config_path="../config"):
            cfg = compose("config", overrides=overrides)
            train.main(cfg)
