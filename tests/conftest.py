import pytest
import pathlib


@pytest.fixture(scope="package")
def root_path():
    return pathlib.Path(__file__).parent.parent.absolute()


@pytest.fixture(scope="package")
def data_path(root_path):
    return root_path.joinpath("data")


@pytest.fixture(scope="package")
def config_path(root_path):
    return root_path.joinpath("config")
