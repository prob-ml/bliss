from pytorch_lightning import Callback
import os
import pytest
from hydra.experimental import initialize, compose

from bliss.hyperparameter import SleepTune


class MetricsCallback(Callback):
    """PyTorch Lightning metric callback."""

    def __init__(self):
        super().__init__()
        self.metrics = []

    def on_validation_end(self, trainer, pl_module):
        self.metrics.append(trainer.callback_metrics)


class TestOptunaSleep:
    def test_optuna_single_core(self, devices):
        use_cuda = devices.use_cuda
        overrides = dict(
            model="basic_sleep_star",
            training="cpu",
            dataset="cpu",
        )
        overrides = [f"{key}={value}" for key, value in overrides.items()]
        with initialize(config_path="../config"):
            cfg = compose("config", overrides=overrides)
            cfg.optimizer.params.update({"lr": 1e-4, "weight_decay": 1e-6})

            if use_cuda:
                # single gpu
                cfg.model.encoder.params.update(
                    {
                        "enc_conv_c": 5,
                        "enc_kern": 3,
                        "enc_hidden": 64,
                    }
                )
                n_epochs = 1

                SleepTune(
                    cfg,
                    max_epochs=n_epochs,
                    model_dir=os.getcwd(),
                    monitor="val_loss",
                    n_batches=4,
                    batch_size=32,
                    n_trials=1,
                    num_gpu=1,
                )
            else:
                cfg.model.encoder.params.update(
                    {
                        "enc_conv_c": 5,
                        "enc_kern": 3,
                        "enc_hidden": 64,
                    }
                )
                n_epochs = 1

                SleepTune(
                    cfg,
                    max_epochs=n_epochs,
                    model_dir=os.getcwd(),
                    monitor="val_loss",
                    n_batches=4,
                    batch_size=32,
                    n_trials=1,
                    num_gpu=0,
                )
