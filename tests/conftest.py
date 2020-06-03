import pytest
import pathlib
import torch

from celeste.datasets.simulated_datasets import get_fitted_powerlaw_psf
from celeste.datasets.galaxy_datasets import DecoderSamples
from celeste import device, use_cuda
from celeste import train
from celeste.datasets import simulated_datasets
from celeste.models import sourcenet


@pytest.fixture(scope="session")
def root_path():
    return pathlib.Path(__file__).parent.parent.absolute()


@pytest.fixture(scope="session")
def data_path(root_path):
    return root_path.joinpath("data")


@pytest.fixture(scope="session")
def config_path(root_path):
    return root_path.joinpath("config")


@pytest.fixture(scope="session")
def single_band_fitted_powerlaw_psf(data_path):
    psf_file = data_path.joinpath("fitted_powerlaw_psf_params.npy")
    return get_fitted_powerlaw_psf(psf_file)[None, 0, ...]


@pytest.fixture(scope="session")
def single_band_galaxy_decoder(data_path):
    galaxy_slen = 51
    n_bands = 1
    galaxy_decoder_file = data_path.joinpath("decoder_params_100_single_band_i.dat")
    return DecoderSamples(galaxy_slen, galaxy_decoder_file, n_bands=n_bands)


@pytest.fixture(scope="session")
def trained_star_encoder(
    config_path, data_path, single_band_galaxy_decoder, fitted_powerlaw_psf
):
    # create training dataset
    n_bands = 2
    max_stars = 20
    mean_stars = 15
    min_stars = 5
    f_min = 1e4
    slen = 50

    # set background
    background = torch.zeros(n_bands, slen, slen, device=device)
    background[0] = 686.0
    background[1] = 1123.0

    # simulate dataset
    n_images = 128
    simulator_args = (
        single_band_galaxy_decoder,
        fitted_powerlaw_psf,
        background,
    )

    simulator_kwargs = dict(
        slen=slen,
        n_bands=n_bands,
        max_sources=max_stars,
        mean_sources=mean_stars,
        min_sources=min_stars,
        f_min=f_min,
        star_prob=1.0,  # enforce only stars are created in the training images.
    )

    dataset = simulated_datasets.SourceDataset(
        n_images, simulator_args, simulator_kwargs
    )

    # setup Star Encoder
    star_encoder = sourcenet.SourceEncoder(
        slen=slen,
        ptile_slen=8,
        step=2,
        edge_padding=3,
        n_bands=n_bands,
        max_detections=2,
        n_source_params=n_bands,  # star has n_bands # fluxes
        enc_conv_c=5,
        enc_kern=3,
        enc_hidden=64,
    ).to(device)

    # train encoder
    # training wrapper
    SleepTraining = train.SleepTraining(
        model=star_encoder,
        dataset=dataset,
        slen=slen,
        n_bands=n_bands,
        n_source_params=n_bands,  # star has n_bands # fluxes
        verbose=False,
        batchsize=32,
    )

    n_epochs = 100 if use_cuda else 1
    SleepTraining.run(n_epochs=n_epochs)

    return star_encoder
