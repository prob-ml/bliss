"""
This script demonstrate how to implement Optuna framework to
achive hyperparameter selection for sleep-phase training.
    Hyperparameters:
        enc_conv_c,
        enc_hidden,
        Learning rate (lr),
        weight_decay
"""

import os, sys

path = os.path.abspath("..")
if path not in sys.path:
    sys.path.insert(0, path)
print(path)
import optuna
import torch
import numpy as np

from bliss.datasets.simulated import SimulatedDataset
from bliss.sleep import Objective


data_path = path + "/data"
device = torch.device("cuda:0")

# get psf
psf_file = data_path + "/fitted_powerlaw_psf_params.npy"
psf_params = torch.from_numpy(np.load(psf_file)).to(device)
psf_params = psf_params[range(1)]

# background
background = torch.zeros(1, 50, 50, device=device)
background[0] = 686.0

# decoder arguments
dec_args = (None, psf_params, background)
dec_kwargs = {}
dec_kwargs.update({"prob_galaxy": 0.0, "n_bands": 1, "slen": 50})

# dataset
star_dataset = SimulatedDataset(4, 32, dec_args, dec_kwargs)

# set up encoder
encoder_kwargs = dict(
    enc_conv_c=(5, 25, 5),
    enc_kern=3,
    enc_hidden=(64, 128, 64),
    ptile_slen=8,
    max_detections=2,
    slen=star_dataset.slen,
    n_bands=star_dataset.n_bands,
    n_galaxy_params=star_dataset.latent_dim,
)

# set up Object for optuna
objects = Objective(
    star_dataset,
    encoder_kwargs,
    max_epochs=100,
    lr=(1e-4, 1e-2),
    weight_decay=(1e-6, 1e-4),
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
