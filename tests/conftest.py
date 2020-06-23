import pytest
import pathlib
import torch

from celeste.models.decoder import get_fitted_powerlaw_psf, get_galaxy_decoder
from celeste import use_cuda, sleep

import pytorch_lightning as pl
from pytorch_lightning.profiler import AdvancedProfiler
from celeste.models import decoder, encoder


def pytest_addoption(parser):
    parser.addoption(
        "--device-id",
        action="store",
        default=0,
        type=int,
        help="ID of cuda device to use.",
    )
    parser.addoption(
        "--profile",
        action="store",
        default=None,
        type=str,
        help="None or file path to store profiler",
    )


@pytest.fixture(scope="session")
def device_id(pytestconfig):
    return pytestconfig.getoption("device_id")


@pytest.fixture(scope="session")
def profile(pytestconfig):
    return pytestconfig.getoption("profile")


@pytest.fixture(scope="session")
def device(device_id):
    new_device = torch.device(f"cuda:{device_id}" if use_cuda else "cpu")
    if use_cuda:
        torch.cuda.set_device(new_device)
    return new_device


@pytest.fixture(scope="session")
def root_path():
    return pathlib.Path(__file__).parent.parent.absolute()


@pytest.fixture(scope="session")
def data_path(root_path):
    return root_path.joinpath("data")


@pytest.fixture(scope="session")
def single_band_fitted_powerlaw_psf(data_path):
    psf_file = data_path.joinpath("fitted_powerlaw_psf_params.npy")
    return get_fitted_powerlaw_psf(psf_file)[None, 0, ...]


@pytest.fixture(scope="session")
def single_band_galaxy_decoder(data_path):

    decoder_state_file = data_path.joinpath("decoder_params_100_single_band_i.dat")
    galaxy_decoder = get_galaxy_decoder(
        decoder_state_file, slen=51, n_bands=1, latent_dim=8
    )
    return galaxy_decoder


@pytest.fixture(scope="session")
def test_star(data_path):
    test_star = torch.load(data_path.joinpath("3_star_test.pt"))
    return test_star


@pytest.fixture(scope="session")
def get_trained_encoder():
    def func(
        galaxy_decoder,
        psf,
        device,
        device_id,
        profile=None,
        n_bands=1,
        max_stars=20,
        mean_stars=15,
        min_stars=5,
        f_min=1e4,
        slen=50,
        n_images=128,
        batch_size=32,
        n_epochs=150,
    ):
        assert galaxy_decoder.n_bands == psf.size(0) == n_bands

        n_epochs = n_epochs if use_cuda else 1

        background = torch.zeros(n_bands, slen, slen, device=device)
        background.fill_(686.0)

        simulator_args = (
            galaxy_decoder,
            psf,
            background,
        )

        simulator_kwargs = dict(
            slen=slen,
            n_bands=n_bands,
            max_sources=max_stars,
            mean_sources=mean_stars,
            min_sources=min_stars,
            f_min=f_min,
            prob_galaxy=0.0,  # only stars will be drawn.
        )

        n_batches = int(n_images / batch_size)
        dataset = decoder.SimulatedDataset(
            n_batches, batch_size, simulator_args, simulator_kwargs
        )

        # setup Star Encoder
        image_encoder = encoder.ImageEncoder(
            slen=slen,
            ptile_slen=8,
            step=2,
            edge_padding=3,
            n_bands=n_bands,
            max_detections=2,
            n_galaxy_params=galaxy_decoder.latent_dim,
            enc_conv_c=5,
            enc_kern=3,
            enc_hidden=64,
        ).to(device)

        # training wrapper
        sleep_net = sleep.SleepPhase(dataset=dataset, image_encoder=image_encoder)

        profiler = AdvancedProfiler(output_filename=profile) if profile else None

        # runs on gpu or cpu?
        n_device = [device_id] if use_cuda else 0  # 0 means no gpu

        sleep_trainer = pl.Trainer(
            gpus=n_device,
            profiler=profiler,
            min_epochs=n_epochs,
            max_epochs=n_epochs,
            reload_dataloaders_every_epoch=True,
        )

        sleep_trainer.fit(sleep_net)

        return sleep_net.image_encoder

    return func
