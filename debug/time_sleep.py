#!/usr/bin/env python
from pathlib import Path
import torch
import numpy as np
import pytorch_lightning as pl
import sys

root_path = Path(".").absolute().parent
sys.path.insert(0, root_path.as_posix())

from bliss.datasets import simulated
from bliss import sleep

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
data_path = root_path.joinpath("data")


@profile
def run_batch():
    # generate batches of data
    for i in range(2):
        dataset.get_batch()


def run_sleep(trainer, model):
    trainer.fit(model)


slen = 50
n_bands = 1

# setup dataset
background = torch.zeros(1, slen, slen, device=device)
background[0] = 5000.0
psf_file = data_path.joinpath("fitted_powerlaw_psf_params.npy")
psf_params = torch.from_numpy(np.load(psf_file)).to(device)[range(n_bands)]

dec_file = data_path.joinpath("galaxy_decoder_1_band.dat")
dec = simulated.SimulatedDataset.get_gal_decoder_from_file(dec_file)
dec_args = (dec, psf_params, background)
dec_kwargs = {
    "prob_galaxy": 1.0,
    "n_bands": n_bands,
    "slen": slen,
    "tile_slen": 2,
    "max_sources_per_tile": 1,
    "mean_sources_per_tile": 0.0020,
    "min_sources_per_tile": 0,
    "ptile_padding": 0,
}

n_batches = 2
batch_size = 2
dataset = simulated.SimulatedDataset(n_batches, batch_size, dec_args, dec_kwargs)

run_batch()

#
# # setup sleep phase
# slen = dataset.slen
# n_bands = dataset.n_bands
# latent_dim = dataset.image_decoder.latent_dim
# n_epochs = 5
# ptile_slen = 8
# tile_slen = 2
# enc_conv_c = 5
# enc_kern = 3
# enc_hidden = 64
# max_detections = 2
#
#
# # setup Star Encoder
# encoder_kwargs = dict(
#     ptile_slen=ptile_slen,
#     tile_slen=tile_slen,
#     enc_conv_c=enc_conv_c,
#     enc_kern=enc_kern,
#     enc_hidden=enc_hidden,
#     max_detections=max_detections,
#     slen=slen,
#     n_bands=n_bands,
#     n_galaxy_params=latent_dim,
# )
#
# sleep_net = sleep.SleepPhase(dataset, encoder_kwargs, validation_plot_start=1000)
#
#
# sleep_trainer = pl.Trainer(
#     gpus="2,",
#     min_epochs=n_epochs,
#     max_epochs=n_epochs,
#     reload_dataloaders_every_epoch=True,
#     logger=False,
#     checkpoint_callback=False,
#     check_val_every_n_epoch=10,
# )
#
# run_sleep(sleep_trainer, sleep_net)
