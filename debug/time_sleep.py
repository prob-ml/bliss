#!/usr/bin/env python
from pathlib import Path
import torch
import numpy as np
import sys

root_path = Path(".").absolute().parent
sys.path.insert(0, root_path.as_posix())

from bliss.datasets import simulated

# cuda setup.
use_cuda = torch.cuda.is_available()
device = torch.device(f"cuda:4" if use_cuda else "cpu")
torch.cuda.set_device(device)

data_path = root_path.joinpath("data")


def run_batch(ds):
    for i in range(100):
        ds.get_batch()


slen = 50
n_bands = 1

# setup dataset
background = torch.zeros(1, slen, slen, device=device)
background[0] = 5000.0
psf_file = data_path.joinpath("fitted_powerlaw_psf_params.npy")
psf_params = torch.from_numpy(np.load(psf_file)).to(device)[range(n_bands)]

# setup decoder
dec_file = data_path.joinpath("galaxy_decoder_1_band.dat")
dec = simulated.SimulatedDataset.get_gal_decoder_from_file(dec_file).to(device)
dec_args = (dec, psf_params, background)

dec_kwargs = {
    "prob_galaxy": 1.0,
    "n_bands": n_bands,
    "slen": slen,
    "tile_slen": 2,
    "ptile_padding": 2,
    "max_sources": 1,
    "mean_sources": 0.0020,
    "min_sources": 0,
}

# setup dataset.
n_batches = 1  # doesn't do anything.
batch_size = 5
dataset = simulated.SimulatedDataset(n_batches, batch_size, dec_args, dec_kwargs)

run_batch(dataset)
