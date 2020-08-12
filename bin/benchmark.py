import os
import sys

path = os.path.abspath("..")
if path not in sys.path:
    sys.path.insert(0, path)

import argparse
import torch
import timeit

from bliss.sleep import SleepPhase
from bliss.wake import WakeNet
from bliss.datasets.simulated import SimulatedDataset
from bliss.models.decoder import get_mgrid


# Benchmark:
## Sleep-phase Dataloader: on GPU 748.68 ms, On CPU 1284.88 ms
## Sleep-phase forward pass: on GPU 1.50 ms, On CPU 2.733 ms

## Wake-phase Dataloader: on GPU 1459.682 ms, On CPU 2137.703 ms
## Wake-phase forward pass: on GPU 4.737 ms, On CPU 5.38 ms


# set up device
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

# sleep phase set up
## set up path
data_path = path + "/data"

## set up Training class
psf_file = data_path + "/fitted_powerlaw_psf_params.npy"
psf_params = SimulatedDataset.get_psf_params_from_file(psf_file).to(device)

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
    tile_slen=2,
    enc_conv_c=5,
    enc_kern=3,
    enc_hidden=64,
    max_detections=2,
    slen=50,
    n_bands=1,
    n_galaxy_params=latent_dim,
)


def benchamark_sleep_setup():
    sleep_net = SleepPhase(dataset, encoder_kwargs)
    return sleep_net.to(device)


# wake phase set up
test_star = torch.load(data_path + "/3_star_test.pt")
test_image = test_star["images"].to(device)
test_slen = test_image.size(-1)

init_background_params = torch.zeros(1, 3, device=device)
init_background_params[0, 0] = 686.0

n_samples = 100
hparams = {"n_samples": n_samples, "lr": 0.001}


def benchmark_wake_setup():
    image_decoder = dataset.image_decoder
    image_decoder.slen = test_slen
    image_decoder.cached_grid = get_mgrid(test_slen)
    sleep_net = benchamark_sleep_setup()
    wake_phase_model = WakeNet(
        sleep_net.image_encoder,
        image_decoder,
        test_image,
        init_background_params,
        hparams,
    )
    return wake_phase_model.to(device)


# add --wake and --sleep argument for users to benchmark different phase
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark Sleep Phase training.")
    parser.add_argument(
        "--sleep",
        "-s",
        action="store_true",
        help="Activate the Bencharmark for the sleep phase training.",
    )

    parser.add_argument(
        "--wake",
        "-w",
        action="store_true",
        help="Activate the Bencharmark for the sleep phase training.",
    )

    args = parser.parse_args()

    # if sleep is specified
    if args.sleep:
        sleep_net = benchamark_sleep_setup()

        def sleep_benchmark():
            with torch.no_grad():
                for batch_idx, batch in enumerate(sleep_net.train_dataloader()):
                    sleep_net.training_step(batch, batch_idx)

        print("Benchmark for the sleep phase training forward pass")
        runtimes = timeit.repeat(
            "sleep_benchmark", repeat=10, number=100, globals=globals(),
        )
        best_time = min(runtimes)
        print(best_time * 1e6, "milliseconds")

    # if wake is specified
    if args.wake:
        wake_phase_model = benchmark_wake_setup()

        def wake_benchmark():
            with torch.no_grad():
                for batch_idx, batch in enumerate(wake_phase_model.train_dataloader()):
                    wake_phase_model.training_step(batch, batch_idx)

        print("Benchmark for the wake phase training forward pass")
        runtimes = timeit.repeat(
            "wake_benchmark", repeat=10, number=200, globals=globals()
        )
        best_time = min(runtimes)
        print(best_time * 1e6, "milliseconds")
