from pytorch_lightning import Callback
from optuna.trial import FixedTrial
from hydra.experimental import initialize, compose

from bliss.sleep import SleepObjective
from bliss.datasets.simulated import SimulatedDataset


class MetricsCallback(Callback):
    """PyTorch Lightning metric callback."""

    def __init__(self):
        super().__init__()
        self.metrics = []

    def on_validation_end(self, trainer, pl_module):
        self.metrics.append(trainer.callback_metrics)


class TestOptunaSleep:
    def test_optuna(self, paths, devices):
        overrides = dict(model="basic_sleep_star", training="cpu", dataset="cpu")
        overrides = [f"{key}={value}" for key, value in overrides.items()]
        with initialize(config_path="../config"):
            cfg = compose("config", overrides=overrides)
            dataset = SimulatedDataset(cfg)
            cfg.model.encoder.params.update(
                {
                    "enc_conv_c": (5, 25, 5),
                    "enc_kern": 3,
                    "enc_hidden": (64, 128, 64),
                    "ptile_slen": 8,
                    "max_detections": 2,
                }
            )
            cfg.optimizer.params.update(
                {"lr": (1e-4, 1e-2), "weight_decay": (1e-6, 1e-4)}
            )
            n_epochs = 1
            # set up Object for optuna
            objects = SleepObjective(
                dataset,
                cfg,
                max_epochs=n_epochs,
                model_dir=paths["model_dir"],
                metrics_callback=MetricsCallback(),
                monitor="val_loss",
                gpus=devices.gpus,
            )

            # set up study object
            objects(
                FixedTrial(
                    {
                        "enc_conv_c": 5,
                        "enc_hidden": 64,
                        "learning rate": 1e-3,
                        "weight_decay": 1e-5,
                    }
                )
            )
