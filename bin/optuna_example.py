#!/usr/bin/env python3
"""
This script demonstrate how to implement Optuna framework to
achive hyperparameter selection for sleep-phase training.
    Hyperparameters:
        enc_conv_c,
        enc_hidden,
        Learning rate (lr),
        weight_decay
"""

import os
import optuna
import hydra
from omegaconf import DictConfig

from pytorch_lightning import Callback
from bliss.hyperparameter import SleepObjective


class MetricsCallback(Callback):
    """PyTorch Lightning metric callback."""

    def __init__(self):
        super().__init__()
        self.metrics = []

    def on_validation_end(self, trainer, pl_module):
        self.metrics.append(trainer.callback_metrics)


def setup_model_dir(args, root_path):
    return root_path.joinpath(args.model_dir)


@hydra.main(config_path="../config", config_name="config")
def main(cfg: DictConfig):
    # grid search
    search_space = {
        "enc_conv_c": [5, 10],
        "enc_hidden": [64, 128],
        "learning rate": [0.001, 0.000492],
        "weight_decay": [1e-6, 0.000003],
    }

    model_dir = os.getcwd()

    monitor = "val_loss"

    pruner = optuna.pruners.MedianPruner(n_startup_trials=30)
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.GridSampler(search_space),
        pruner=pruner,
    )

    cfg.model.encoder.params.update(
        {
            "enc_conv_c": (5, 25, 5),
            "enc_kern": 3,
            "enc_hidden": (64, 128, 64),
            "ptile_slen": 8,
            "max_detections": 2,
        }
    )
    print(cfg.model.encoder.params.enc_conv_c)
    cfg.optimizer.params.update({"lr": (1e-4, 1e-2), "weight_decay": (1e-6, 1e-4)})
    objects = SleepObjective(
        cfg,
        max_epochs=100,
        model_dir=model_dir,
        metrics_callback=MetricsCallback(),
        monitor=monitor,
        n_batches=4,
        batch_size=32,
    )

    study.optimize(
        objects,
        n_trials=100,
        timeout=600,
    )


if __name__ == "__main__":
    main()
