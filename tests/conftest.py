import pytest
import pathlib
import torch

from celeste.datasets.simulated_datasets import get_fitted_powerlaw_psf
from celeste.datasets.galaxy_datasets import DecoderSamples
from celeste import use_cuda


def pytest_addoption(parser):
    parser.addoption(
        "--device-id", action="store", default=0, help="ID of cuda device to use."
    )


@pytest.fixture(scope="session")
def device_id(pytestconfig):
    return pytestconfig.getoption("device_id")


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
    galaxy_slen = 51
    n_bands = 1
    galaxy_decoder_file = data_path.joinpath("decoder_params_100_single_band_i.dat")
    return DecoderSamples(galaxy_slen, galaxy_decoder_file, n_bands=n_bands)


@pytest.fixture(scope="session")
def test_star(data_path):
    test_star = torch.load(data_path.joinpath("3_star_test.pt"))
    return test_star


def pytest_addoption(parser):
    parser.addoption(
        "--profile",
        action="store",
        default=None,
        help="None or file path to store profiler",
    )


@pytest.fixture(scope="session")
def profile(pytestconfig):
    return pytestconfig.getoption("--profile")
