#!/usr/bin/env python3

import pathlib
import torch
import timeit
import gc

from bliss import sleep
from bliss.datasets.simulated import SimulatedDataset
from bliss.models import encoder


root_path = pathlib.Path(__file__).parent.parent.absolute()
data_path = root_path.joinpath("data")

psf_file = data_path.joinpath("fitted_powerlaw_psf_params.npy")
psf = SimulatedDataset.get_psf_from_file(psf_file)
psf = torch.unsqueeze(psf[0], 0)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
background = torch.zeros(1, 50, 50, device=device)
background[0] = 686.0

dec_args = (None, psf, background)

dataset = SimulatedDataset(
    n_batches=1,
    batch_size=32,
    decoder_args=dec_args,
    decoder_kwargs={"prob_galaxy": 0.0, "n_bands": 1, "slen": 50},
)

latent_dim = dataset.image_decoder.latent_dim
image_encoder = encoder.ImageEncoder(
    slen=50,
    ptile_slen=8,
    step=2,
    edge_padding=3,
    n_bands=1,
    max_detections=2,
    n_galaxy_params=latent_dim,
    enc_conv_c=5,
    enc_kern=3,
    enc_hidden=64,
).to(device)

sleep_net = sleep.SleepPhase(dataset=dataset, image_encoder=image_encoder)

print(
    timeit.Timer("sleep_net.train_dataloader", "gc.enable()", globals=globals()).repeat(
        repeat=5, number=1
    )
)


def single_forward():
    with torch.no_grad():
        for batch in sleep_net.train_dataloader():
            sleep_net.training_step(batch)


print(
    timeit.Timer("single_forward", "gc.enable()", globals=globals()).repeat(
        repeat=5, number=1
    )
)
