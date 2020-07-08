import pytest
import pathlib
import torch
from pytorch_lightning.profiler import AdvancedProfiler

from bliss import use_cuda
from bliss.datasets.simulated import SimulatedDataset


def pytest_addoption(parser):
    parser.addoption(
        "--device-id",
        action="store",
        default=0,
        type=int,
        help="ID of cuda device to use.",
    )

    parser.addoption(
        "--gpus", default="0,", type=str, help="--gpus option for trainer."
    )

    parser.addoption(
        "--profile", action="store_true", help="Enable profiler",
    )

    parser.addoption(
        "--log", action="store_true", help="Enable logger.",
    )

    parser.addoption(
        "--repeat", default=1, type=str, help="Number of times to repeat each test"
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


# paths
@pytest.fixture(scope="session")
def root_path():
    return pathlib.Path(__file__).parent.parent.absolute()


@pytest.fixture(scope="session")
def logs_path(root_path):
    logs_path = root_path.joinpath("tests/logs")
    logs_path.mkdir(exist_ok=True, parents=True)
    return logs_path


@pytest.fixture(scope="session")
def data_path(root_path):
    return root_path.joinpath("data")


# logging and profiling.
@pytest.fixture(scope="session")
def profiler(pytestconfig, logs_path):
    profiling = pytestconfig.getoption("profile")
    profile_file = logs_path.joinpath("profile.txt")
    profiler = AdvancedProfiler(output_filename=profile_file) if profiling else None
    return profiler


@pytest.fixture(scope="session")
def save_logs(pytestconfig):
    return pytestconfig.getoption("log")


# data and memory.
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
def gpus(pytestconfig):
    if use_cuda:
        return pytestconfig.getoption("gpus")
    else:
        return None


@pytest.fixture(scope="session")
def star_dataset(data_path, device):
    psf_file = data_path.joinpath("fitted_powerlaw_psf_params.npy")
    decoder_state_file = data_path.joinpath("decoder_params_100_single_band_i.dat")

    n_bands = 1
    slen = 50
    max_stars = 20
    mean_stars = 15
    min_stars = 5
    f_min = 1e4
    n_images = 128
    batch_size = 32
    prob_galaxy = 0.0
    n_batches = int(n_images / batch_size)

    psf = SimulatedDataset.get_psf_from_file(psf_file)[None, 0, ...]
    galaxy_decoder = SimulatedDataset.get_gal_decoder_from_file(
        decoder_state_file, slen=51, n_bands=1, latent_dim=8
    )
    background = torch.zeros(n_bands, slen, slen, device=device)
    background.fill_(686.0)

    decoder_args = (
        galaxy_decoder,
        psf,
        background,
    )

    decoder_kwargs = {
        "slen": slen,
        "n_bands": n_bands,
        "max_sources": max_stars,
        "mean_sources": mean_stars,
        "min_sources": min_stars,
        "f_min": f_min,
        "prob_galaxy": prob_galaxy,
    }

    assert galaxy_decoder.n_bands == psf.size(0) == n_bands
    star_dataset = SimulatedDataset(n_batches, batch_size, decoder_args, decoder_kwargs)

    return star_dataset


@pytest.fixture(scope="session")
def test_3_stars(data_path):
    test_star = torch.load(data_path.joinpath("3_star_test.pt"))
    return test_star
