import shutil

import pytest
import torch
from hydra import compose, initialize

from bliss import tune


class TestTune:
    @pytest.fixture(scope="class")
    def overrides(self, devices, paths):
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
            "dataset.kwargs.n_batches": 10 if devices.use_cuda else 2,
            "training": "m2" if devices.use_cuda else "cpu",
            "optimizer": "m2",
            "tuning.n_epochs": 2 if devices.use_cuda else 1,
            "tuning.allocated_gpus": allocated_gpus,
            "tuning.gpus_per_trial": gpus_per_trial,
            "tuning.grace_period": 1,
            "tuning.verbose": 0,
            "tuning.save": False,
            "tuning.n_samples": 2 if devices.use_cuda else 1,
            "tuning.log_path": f"{paths['root']}/tuning",
        }
        return [f"{k}={v}" for k, v in overrides.items()]

    @pytest.mark.filterwarnings("ignore:.*Relying on `self.log.*:DeprecationWarning")
    def test_tune_run(self, paths, overrides):
        with initialize(config_path="../config"):
            cfg = compose("config", overrides=overrides)
            tune.tune(cfg)
        shutil.rmtree(f"{paths['root']}/tuning")
