import shutil

import pytest
import torch

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
            "tuning.model": "${models.sleep}",
            "tuning.n_epochs": 2 if devices.use_cuda else 1,
            "tuning.allocated_gpus": allocated_gpus,
            "tuning.gpus_per_trial": gpus_per_trial,
            "tuning.grace_period": 1,
            "tuning.verbose": 0,
            "tuning.save": False,
            "tuning.n_samples": 2 if devices.use_cuda else 1,
            "tuning.log_path": "${paths.root}/tuning",
        }

        if not devices.use_cuda:
            overrides.update(
                {
                    "datasets.simulated_m2.n_batches": 1,
                    "datasets.simulated_m2.batch_size": 2,
                    "datasets.simulated_m2.generate_device": "cpu",
                    "datasets.simulated_m2.testing_file": None,
                }
            )
        return overrides

    @pytest.mark.filterwarnings("ignore:.*Relying on `self.log.*:DeprecationWarning")
    def test_tune_run(self, overrides, m2_model_setup):
        cfg = m2_model_setup.get_cfg(overrides)
        tune.tune(cfg)
        shutil.rmtree(f"{cfg.paths.root}/tuning")
