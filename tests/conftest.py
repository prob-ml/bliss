import pytest
import pathlib
from celeste.datasets.simulated_datasets import get_fitted_powerlaw_psf


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
def fitted_powerlaw_psf(data_path):
    psf_file = data_path.joinpath("fitted_powerlaw_psf_params.npy")
    return get_fitted_powerlaw_psf(psf_file)
