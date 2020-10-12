import pytest
import pathlib
import torch
import pytorch_lightning as pl

from bliss import sleep
from bliss.datasets.simulated import SimulatedDataset


# command line arguments for tests
def pytest_addoption(parser):

    parser.addoption(
        "--gpus", default="0,", type=str, help="--gpus option for trainer."
    )

    parser.addoption(
        "--repeat", default=1, type=str, help="Number of times to repeat each test"
    )

    parser.addoption(
        "--runslow", action="store_true", default=False, help="run slow tests"
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: mark test as slow to run")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--runslow"):
        # --runslow given in cli: do not skip slow tests
        return
    skip_slow = pytest.mark.skip(reason="need --runslow option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)


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


class DeviceSetup:
    def __init__(self, gpus):
        self.use_cuda = torch.cuda.is_available()
        self.gpus = gpus if self.use_cuda else None

        # setup device
        self.device = torch.device("cpu")
        if self.gpus and self.use_cuda:
            device_id = self.gpus.split(",")
            assert len(device_id) == 2 and device_id[1] == ""
            device_id = int(self.gpus[0])
            self.device = torch.device(f"cuda:{device_id}")
            torch.cuda.set_device(self.device)


class EncoderSetup:
    def __init__(self, gpus, device):
        self.gpus = gpus
        self.device = device

    def get_trained_encoder(
        self,
        dataset,
        n_epochs=100,
        ptile_slen=8,
        tile_slen=2,
        max_detections=2,
        enc_hidden=256,
        enc_kern=3,
        enc_conv_c=20,
        validation_plot_start=1000,
        background_pad_value=686.0,
    ):
        slen = dataset.slen
        n_bands = dataset.n_bands
        latent_dim = dataset.image_decoder.latent_dim

        # setup Star Encoder
        encoder_kwargs = dict(
            ptile_slen=ptile_slen,
            tile_slen=tile_slen,
            enc_conv_c=enc_conv_c,
            enc_kern=enc_kern,
            enc_hidden=enc_hidden,
            max_detections=max_detections,
            slen=slen,
            n_bands=n_bands,
            n_galaxy_params=latent_dim,
            background_pad_value=background_pad_value,
        )

        sleep_net = sleep.SleepPhase(
            dataset, encoder_kwargs, validation_plot_start=validation_plot_start
        )

        sleep_trainer = pl.Trainer(
            gpus=self.gpus,
            min_epochs=n_epochs,
            max_epochs=n_epochs,
            reload_dataloaders_every_epoch=True,
            logger=False,
            checkpoint_callback=False,
            check_val_every_n_epoch=50,
        )

        sleep_trainer.fit(sleep_net)
        sleep_net.image_encoder.eval()
        return sleep_net.image_encoder.to(self.device)


# available fixtures
@pytest.fixture(scope="session")
def paths():
    root_path = pathlib.Path(__file__).parent.parent.absolute()
    return {
        "root": root_path,
        "data": root_path.joinpath("data"),
        "model_dir": root_path.joinpath("trials_result"),
    }


@pytest.fixture(scope="session")
def device_setup(pytestconfig):
    gpus = pytestconfig.getoption("gpus")
    return DeviceSetup(gpus)


@pytest.fixture(scope="session")
def decoder_setup(paths, device_setup):
    return DecoderSetup(paths, device_setup.device)


@pytest.fixture(scope="session")
def encoder_setup(device_setup):
    return EncoderSetup(device_setup.gpus, device_setup.device)
