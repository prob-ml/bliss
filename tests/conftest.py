import pytest
import pathlib
import torch
import numpy as np

from celeste import device
from celeste import psf_transform


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
    psf_params = torch.from_numpy(np.load(psf_file)).to(device)
    power_law_psf = psf_transform.PowerLawPSF(psf_params)
    psf = power_law_psf.forward().detach()
    assert psf.size(0) == 2 and psf.size(1) == psf.size(2) == 101
    return psf
