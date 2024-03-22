# pylint: skip-file

import pytest
import torch


# command line arguments for tests
def pytest_addoption(parser):
    parser.addoption(
        "--gpu",
        action="store_true",
        default=False,
        help="Run tests using gpu.",
    )


class DeviceSetup:
    def __init__(self, use_gpu):
        self.use_cuda = torch.cuda.is_available() if use_gpu else False
        self.accelerator = "gpu" if self.use_cuda else "cpu"
        self.device = torch.device("cpu")
        if self.use_cuda:
            self.device = torch.device("cuda:0")
            torch.cuda.set_device(self.device)


@pytest.fixture(scope="session")
def devices(pytestconfig):
    use_gpu = pytestconfig.getoption("gpu")
    return DeviceSetup(use_gpu)
