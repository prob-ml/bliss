import os, sys

path = os.path.abspath("..")
if path not in sys.path:
    sys.path.insert(0, path)
print(path)
import optuna
import torch
import numpy as np

import multiprocessing as mp

from pytorch_lightning import Callback

from bliss.hyperparameter import SleepObjective


class MetricsCallback(Callback):
    """PyTorch Lightning metric callback."""

    def __init__(self):
        super().__init__()
        self.metrics = []

    def on_validation_end(self, trainer, pl_module):
        self.metrics.append(trainer.callback_metrics)


metrics_callback = MetricsCallback()

model_dir = os.getcwd()
data_path = path + "/data"

# get psf
psf_file = path + "/data/fitted_powerlaw_psf_params.npy"
psf_params = torch.from_numpy(np.load(psf_file))
psf_params = psf_params[range(1)]

# background
background = torch.zeros(1, 50, 50)
background[0] = 686.0

# decoder arguments
dec_args = (None, psf_params, background)
dec_kwargs = {}
dec_kwargs.update({"prob_galaxy": 0.0, "n_bands": 1, "slen": 50})

# set up encoder
encoder_kwargs = dict(
    enc_conv_c=(5, 25, 5),
    enc_kern=3,
    enc_hidden=(64, 128, 64),
    ptile_slen=8,
    max_detections=2,
    slen=50,
    n_bands=1,
    n_galaxy_params=8,
)

monitor = "val_loss"

sleepobjectiveargs = {
    "encoder_kwargs": encoder_kwargs,
    "max_epochs": 100,
    "lr": (1e-4, 1e-2),
    "weight_decay": (1e-6, 1e-4),
    "model_dir": model_dir,
    "metrics_callback": metrics_callback,
    "monitor": monitor,
    "n_batches": 4,
    "batch_size": 32,
    "dec_args": dec_args,
    "dec_kwargs": dec_kwargs,
}

if __name__ == "__main__":
    # grid search
    search_space = {
        "enc_conv_c": [5, 10],
        "enc_hidden": [64, 128],
        "learning rate": [0.001, 0.000492],
        "weight_decay": [1e-6, 0.000003],
    }

    pruner = optuna.pruners.MedianPruner(n_startup_trials=30)
    study = optuna.create_study(
        storage="sqlite:///zz.db",
        direction="minimize",
        sampler=optuna.samplers.GridSampler(search_space),
        pruner=pruner,
    )
    processes = []
    n_gpu = (1, 4, 6)

    # set up queue of devices
    mpspawn = mp.get_context("spawn")
    gpu_queue = mpspawn.Manager().Queue()
    for i in n_gpu:
        gpu_queue.put(i)

    sleepobjectiveargs["gpu_queue"] = gpu_queue

    study.optimize(
        SleepObjective(**sleepobjectiveargs),
        n_trials=100,
        n_jobs=len(n_gpu),
        timeout=600,
    )
