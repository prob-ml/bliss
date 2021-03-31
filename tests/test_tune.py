import pytest
import torch
from hydra.experimental import initialize, compose
from bliss import tune


class TestTune:
    @pytest.fixture(scope="class")
    def overrides(self, devices):
        allocated_gpus = 0
        gpus_per_trial = 0
        if devices.use_cuda:
            gpus_per_trial = 1
            if torch.cuda.device_count() >= 2:
                allocated_gpus = 2
            else:
                allocated_gpus = 1

        overrides = {
            "model": "m2",
            "dataset": "m2" if devices.use_cuda else "cpu",
            "dataset.params.n_batches": 2 if devices.use_cuda else 10,
            "training": "m2" if devices.use_cuda else "cpu",
            "optimizer": "m2",
            "tuning.n_epochs": 2 if devices.use_cuda else 1,
            "tuning.allocated_gpus": allocated_gpus,
            "tuning.gpus_per_trial": gpus_per_trial,
            "tuning.grace_period": 1,
            "tuning.verbose": 0,
            "tuning.save": False,
            "tuning.n_samples": 2 if devices.use_cuda else 1,
        }
        overrides = [f"{k}={v}" for k, v in overrides.items()]
        return overrides

    @pytest.mark.filterwarnings("ignore:.*PytestUnhandledThreadExceptionWarning.*")
    @pytest.mark.filterwarnings("ignore:.*Relying on `self.log.*:DeprecationWarning")
    def test_tune_run(self, overrides, devices):
        with initialize(config_path="../config"):
            cfg = compose("config", overrides=overrides)
            tune.tune(cfg, local_mode=(not devices.use_cuda))
