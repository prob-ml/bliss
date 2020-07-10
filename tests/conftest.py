import pytest
import pathlib
import torch
import pytorch_lightning as pl
from pytorch_lightning.profiler import AdvancedProfiler

from bliss import use_cuda, sleep
from bliss.datasets.simulated import SimulatedDataset
from bliss.models import galaxy_net, encoder


def pytest_addoption(parser):

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


@pytest.fixture(scope="session")
def gpus(pytestconfig):
    gpus = pytestconfig.getoption("gpus")
    if not use_cuda:
        gpus = None

    return gpus


@pytest.fixture(scope="session")
def device(gpus):
    device_id = gpus.split(",")
    assert len(device_id) == 2 and device_id[1] == ""
    device_id = int(gpus[0])
    new_device = torch.device(f"cuda:{device_id}" if use_cuda else "cpu")
    if use_cuda:
        torch.cuda.set_device(new_device)
    return new_device


@pytest.fixture(scope="session")
def galaxy_decoder(data_path):
    slen = 51
    latent_dim = 8
    n_bands = 1
    decoder_file = data_path.joinpath("decoder_params_100_single_band_i.dat")
    dec = galaxy_net.CenteredGalaxyDecoder(slen, latent_dim, n_bands).to(device)
    dec.load_state_dict(torch.load(decoder_file, map_location=device))
    dec.eval()
    return dec


@pytest.fixture(scope="session")
def fitted_psf(data_path):
    psf_file = data_path.joinpath("fitted_powerlaw_psf_params.npy")
    psf = SimulatedDataset.get_psf_from_file(psf_file)
    assert psf.size(0) == 2
    assert len(psf.shape) == 3
    return psf


@pytest.fixture(scope="session")
def get_star_dataset(device):
    def star_dataset(
        psf, batch_size=32, n_images=128, n_bands=1, slen=50, **dec_kwargs
    ):
        assert 1 <= n_bands <= 2
        assert len(psf.shape) == 3

        dec_kwargs.update({"prob_galaxy": 0.0, "n_bands": n_bands, "slen": slen})
        background = torch.zeros(2, slen, slen, device=device)
        background[0] = 686.0
        background[1] = 1123.0

        # slice if necessary.
        background = background[range(n_bands)]
        psf = psf[range(n_bands)]

        dec_args = (None, psf, background)

        n_batches = int(n_images / batch_size)
        return SimulatedDataset(n_batches, batch_size, dec_args, dec_kwargs)

    return star_dataset


@pytest.fixture(scope="session")
def get_trained_star_encoder(device, gpus, profiler, save_logs, logs_path):
    def trained_star_encoder(
        star_dataset, n_epochs=100, enc_conv_c=5, enc_kern=3, enc_hidden=64
    ):
        n_epochs = n_epochs if use_cuda else 1

        slen = star_dataset.slen
        n_bands = star_dataset.n_bands
        latent_dim = star_dataset.image_decoder.latent_dim

        # setup Star Encoder
        image_encoder = encoder.ImageEncoder(
            slen=slen,
            ptile_slen=8,
            step=2,
            edge_padding=3,
            n_bands=n_bands,
            max_detections=2,
            n_galaxy_params=latent_dim,
            enc_conv_c=enc_conv_c,
            enc_kern=enc_kern,
            enc_hidden=enc_hidden,
        ).to(device)

        sleep_net = sleep.SleepPhase(
            dataset=star_dataset, image_encoder=image_encoder, save_logs=save_logs
        )

        sleep_trainer = pl.Trainer(
            gpus=gpus,
            profiler=profiler,
            min_epochs=n_epochs,
            max_epochs=n_epochs,
            reload_dataloaders_every_epoch=True,
            default_root_dir=logs_path,
        )

        sleep_trainer.fit(sleep_net)
        sleep_net.image_encoder.eval()
        return sleep_net.image_encoder

    return trained_star_encoder


@pytest.fixture(scope="session")
def test_3_stars(data_path):
    test_star = torch.load(data_path.joinpath("3_star_test.pt"))
    return test_star
