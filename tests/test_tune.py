import pytest
from hydra.experimental import initialize, compose
from bliss import tune


class TestTune:
    @pytest.fixture(scope="class")
    def overrides(self, devices):
        overrides = {
            "model": "m2",
            "dataset": "m2" if devices.use_cuda else "cpu",
            "dataset.params.n_batches": 2 if devices.use_cuda else 10,
            "training": "m2" if devices.use_cuda else "cpu",
            "optimizer": "m2",
            "tuning.n_epochs": 2 if devices.use_cuda else 1,
            "tuning.allocated_gpus": 2 if devices.use_cuda else 0,
            "tuning.gpus_per_trial": 1 if devices.use_cuda else 0,
            "tuning.max_concurrent": 1,
            "tuning.grace_period": 1,
            "tuning.verbose": 0,
            "tuning.save": False,
            "tuning.n_samples": 2 if devices.use_cuda else 1,
        }
        overrides = [f"{k}={v}" for k, v in overrides.items()]
        return overrides

    def test_tune_run(self, overrides, devices):
        with initialize(config_path="../config"):
            cfg = compose("config", overrides=overrides)
            tune.main(cfg, local_mode=(not devices.use_cuda))
