import pytest
import pathlib
import torch

from celeste.datasets.simulated_datasets import get_fitted_powerlaw_psf
from celeste.datasets.galaxy_datasets import DecoderSamples


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
