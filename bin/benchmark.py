#!/usr/bin/env python3

import pathlib
import torch
from line_profiler import LineProfiler

from bliss import sleep, wake
from bliss.datasets.simulated import SimulatedDataset
from bliss.models.decoder import get_mgrid


# set up path
root_path = pathlib.Path(__file__).parent.parent.absolute()
data_path = root_path.joinpath("data")

# set up Training class
psf_file = data_path.joinpath("fitted_powerlaw_psf_params.npy")
psf_params = SimulatedDataset.get_psf_params_from_file(psf_file)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
background = torch.zeros(1, 50, 50, device=device)
background[0] = 686.0

dec_args = (None, psf_params[range(1)], background)

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

# profile the dataloader
sleep_profile = LineProfiler(sleep_net.train_dataloader)
sleep_profile.runcall(sleep_net.train_dataloader)


# profile the forward step
with torch.no_grad():
    for batch_idx, batch in enumerate(sleep_net.train_dataloader()):
        sleep_profile.add_function(sleep_net.training_step)
        sleep_profile.runcall(sleep_net.training_step, batch, batch_idx)
        sleep_profile.print_stats()

# save the profile
sleep_profile.dump_stats("sleep_dataloader_and_forward")


# Profile the wake face
## set up
test_star = torch.load(data_path.joinpath("3_star_test.pt"))
test_image = test_star["images"]
test_slen = test_image.size(-1)

init_background_params = torch.zeros(1, 3, device=device)
init_background_params[0, 0] = 686.0

n_samples = 100
hparams = {"n_samples": n_samples, "lr": 0.001}

image_decoder = dataset.image_decoder
image_decoder.slen = test_slen
image_decoder.cached_grid = get_mgrid(test_slen)
wake_phase_model = wake.WakeNet(
    sleep_net.image_encoder, image_decoder, test_image, init_background_params, hparams,
)

# Profile wake phase dataloader
wake_profile = LineProfiler(wake_phase_model.train_dataloader)
wake_profile.runcall(wake_phase_model.train_dataloader)

# Profile wake phase forward
with torch.no_grad():
    for batch_idx, batch in enumerate(wake_phase_model.train_dataloader()):
        wake_profile.add_function(wake_phase_model.training_step)
        wake_profile.runcall(wake_phase_model.training_step, batch, batch_idx)
        wake_profile.print_stats()

# save the profile
wake_profile.dump_stats("wake_dataloader_and_forward")
