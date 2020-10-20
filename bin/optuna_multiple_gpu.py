import os
import optuna
import hydra
import warnings
from omegaconf import DictConfig

import multiprocessing as mp
from joblib import parallel_backend

from pytorch_lightning import Callback

from bliss.hyperparameter import SleepObjective


class MetricsCallback(Callback):
    """PyTorch Lightning metric callback."""

    def __init__(self):
        super().__init__()
        self.metrics = []

    def on_validation_end(self, trainer, pl_module):
        self.metrics.append(trainer.callback_metrics)


@hydra.main(config_path="../config", config_name="config")
def main(cfg: DictConfig):

    model_dir = os.getcwd()

    monitor = "val_loss"

    pruner = optuna.pruners.MedianPruner(n_startup_trials=30)
    study = optuna.create_study(
        storage="sqlite:///zz.db",
        direction="minimize",
        pruner=pruner,
    )
    n_gpu = (1, 3, 5)

    # set up queue of devices
    mp.set_start_method("spawn")
    gpu_queue = mp.Manager().Queue()
    for i in n_gpu:
        gpu_queue.put(i)

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
        gpu_queue=gpu_queue,
    )

    with parallel_backend("multiprocessing", n_jobs=len(n_gpu)):
        warnings.simplefilter("error")
        study.optimize(
            objects,
            n_trials=100,
            n_jobs=len(n_gpu),
            timeout=600,
        )


if __name__ == "__main__":
    main()
