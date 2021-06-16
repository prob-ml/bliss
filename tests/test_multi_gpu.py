import pytest
import torch
from hydra import initialize, compose
from bliss import train


@pytest.mark.multi_gpu
@pytest.mark.filterwarnings(
    "ignore:You requested multiple GPUS but did not specify a backend.*:UserWarning"
)
def test_train_run(devices):
    if devices.use_cuda:
        overrides = {
            "dataset.kwargs.n_batches": 5,
            "training.trainer.logger": "False",
            "training.n_epochs": 5,
            "gpus": min(3, torch.cuda.device_count()),
            "+training.trainer.accelerator": "ddp",
        }
        overrides = [f"{k}={v}" for k, v in overrides.items()]
        with initialize(config_path="../config"):
            cfg = compose("config", overrides=overrides)
            train.train(cfg)
