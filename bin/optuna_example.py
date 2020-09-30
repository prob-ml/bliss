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

import argparse
import optuna
import torch
import numpy as np

from .utils import setup_paths, add_path_args

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


def main(args):

    # setup
    paths = setup_paths(args, enforce_overwrite=False)
    metrics_callback = MetricsCallback()

    # get psf
    psf_file = paths["data"].joinpath("fitted_powerlaw_psf_params.npy")
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

    model_dir = setup_model_dir(args, paths["root"])
    monitor = args.monitor

    # set up Object for optuna
    objects = SleepObjective(
        encoder_kwargs,
        max_epochs=100,
        lr=(1e-4, 1e-2),
        weight_decay=(1e-6, 1e-4),
        model_dir=model_dir,
        metrics_callback=metrics_callback,
        monitor=monitor,
        n_batches=4,
        batch_size=32,
        dec_args=dec_args,
        dec_kwargs=dec_kwargs,
    )

    # use pruner
    pruner = optuna.pruners.MedianPruner()

    # set up study object
    study = optuna.create_study(direction="minimize", pruner=pruner)
    study.optimize(objects, n_trials=100, timeout=600)

    # print out the best result
    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="set up the saving path for hyperparameter selection under optuna framework",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Paths
    parser = add_path_args(parser)

    # model_dir
    parser.add_argument(
        "--model_dir", type=str, required=True, help="what path to store logs"
    )

    # monitor
    parser.add_argument(
        "--monitor",
        type=str,
        required=True,
        help="which value from validation logs to monitor for hyperparameter selection",
    )

    pargs = parser.parse_args()
    main(pargs)
