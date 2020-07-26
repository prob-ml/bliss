#!/usr/bin/env python3

import pathlib
import torch
import timeit
from line_profiler import LineProfiler

from bliss import sleep
from bliss.datasets.simulated import SimulatedDataset
from bliss.models import encoder


root_path = pathlib.Path(__file__).parent.parent.absolute()
data_path = root_path.joinpath("data")

psf_file = data_path.joinpath("fitted_powerlaw_psf_params.npy")
psf_params = SimulatedDataset.get_psf_params_from_file(psf_file)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
background = torch.zeros(1, 50, 50, device=device)
background[0] = 686.0

dec_args = (None, psf_params, background)

dataset = SimulatedDataset(
    n_batches=1,
    batch_size=32,
    decoder_args=dec_args,
    decoder_kwargs={"prob_galaxy": 0.0, "n_bands": 1, "slen": 50},
)

latent_dim = dataset.image_decoder.latent_dim

encoder_kwargs = dict(
    ptile_slen=8,
    step=2,
    edge_padding=3,
    enc_conv_c=5,
    enc_kern=3,
    enc_hidden=64,
    max_detections=2,
    slen=50,
    n_bands=1,
    n_galaxy_params=latent_dim,
)

sleep_net = sleep.SleepPhase(dataset, encoder_kwargs)

profile = LineProfiler(sleep_net.train_dataloader)
profile.runcall(sleep_net.train_dataloader)
profile.print_stats()


with torch.no_grad():
    for batch_idx, batch in enumerate(sleep_net.train_dataloader()):
        profile.add_function(sleep_net.training_step)
        profile.runcall(sleep_net.training_step, batch, batch_idx)
        profile.print_stats()
