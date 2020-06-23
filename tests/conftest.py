import pytest
import pathlib
import torch

from celeste.models.decoder import get_fitted_powerlaw_psf, get_galaxy_decoder
from celeste import use_cuda


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
