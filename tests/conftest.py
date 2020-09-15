import pytest
import pathlib
import torch
import numpy as np
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


class DecoderSetup:
    def __init__(self, paths, device):
        self.device = device
        self.data_path = paths["data"]

    def get_galaxy_decoder(self):
        dec_file = self.data_path.joinpath("galaxy_decoder_1_band.dat")
        dec = SimulatedDataset.get_gal_decoder_from_file(dec_file)
        return dec

    def get_fitted_psf_params(self):
        psf_file = self.data_path.joinpath("fitted_powerlaw_psf_params.npy")
        psf_params = torch.from_numpy(np.load(psf_file)).to(self.device)
        return psf_params

    def get_star_dataset(
        self,
        init_psf_params,
        batch_size=32,
        n_batches=4,
        n_bands=1,
        slen=50,
        **dec_kwargs,
    ):
        assert 1 <= n_bands <= 2
        dec_kwargs.update({"prob_galaxy": 0.0, "n_bands": n_bands, "slen": slen})
        background = torch.zeros(2, slen, slen, device=self.device)
        background[0] = 686.0
        background[1] = 1123.0

        # slice if necessary.
        background = background[list(range(n_bands))]
        init_psf_params = init_psf_params[range(n_bands)]

        dec_args = (None, init_psf_params, background)

        return SimulatedDataset(n_batches, batch_size, dec_args, dec_kwargs)

    def get_binary_dataset(
        self, batch_size=32, n_batches=4, slen=20, prob_galaxy=1.0, **dec_kwargs
    ):

        n_bands = 1
        galaxy_decoder = self.get_galaxy_decoder()

        # psf params
        psf_params = self.get_fitted_psf_params()[list(range(n_bands))]

        # TODO: take background from test image.
        background = torch.zeros(n_bands, slen, slen, device=self.device)
        background[0] = 5000.0

        dec_args = (galaxy_decoder, psf_params, background)

        dec_kwargs.update(
            {"n_bands": n_bands, "slen": slen, "prob_galaxy": prob_galaxy}
        )

        return SimulatedDataset(n_batches, batch_size, dec_args, dec_kwargs)


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
