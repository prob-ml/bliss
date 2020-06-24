import pytest
import pathlib
import torch

from celeste import use_cuda
from celeste.models.decoder import get_fitted_powerlaw_psf, get_galaxy_decoder


def pytest_addoption(parser):
    parser.addoption(
        "--device-id",
        action="store",
        default=0,
        type=int,
        help="ID of cuda device to use.",
    )
    parser.addoption(
        "--sprof",
        action="store",
        default=None,
        type=str,
        help="None or file path to store sleep phase training profiler",
    )
    parser.addoption(
        "--wprof",
        action="store",
        default=None,
        type=str,
        help="None or file path to store wake phase training profiler",
    )
    parser.addoption(
        "--log",
        action="store",
        default=False,
        type=bool,
        help="False or True to enable logger for the training",
    )

    parser.addoption(
        "--repeat", action="store", help="Number of times to repeat each test"
    )


# allows test repetition with --repeat flag.
def pytest_generate_tests(metafunc):
    if metafunc.config.option.repeat is not None:
        count = int(metafunc.config.option.repeat)

        # We're going to duplicate these tests by parametrizing them,
        # which requires that each test has a fixture to accept the parameter.
        # We can add a new fixture like so:
        metafunc.fixturenames.append("tmp_ct")

        # Now we parametrize. This is what happens when we do e.g.,
        # @pytest.mark.parametrize('tmp_ct', range(count))
        # def test_foo(): pass
        metafunc.parametrize("tmp_ct", range(count))


@pytest.fixture(scope="session")
def device_id(pytestconfig):
    return pytestconfig.getoption("device_id")


@pytest.fixture(scope="session")
def sprof(pytestconfig):
    return pytestconfig.getoption("sprof")


@pytest.fixture(scope="session")
def wprof(pytestconfig):
    return pytestconfig.getoption("wprof")


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
def test_3_stars(data_path):
    test_star = torch.load(data_path.joinpath("3_star_test.pt"))
    return test_star
