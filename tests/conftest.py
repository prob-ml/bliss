import pytest
import pathlib
import torch
import numpy as np
import pytorch_lightning as pl

from bliss import use_cuda, sleep
from bliss.datasets.simulated import SimulatedDataset


def pytest_addoption(parser):

    parser.addoption(
        "--gpus", default="0,", type=str, help="--gpus option for trainer."
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
def data_path(root_path):
    return root_path.joinpath("data")


@pytest.fixture(scope="session")
def gpus(pytestconfig):
    gpus = pytestconfig.getoption("gpus")
    if not use_cuda:
        gpus = None

    return gpus


@pytest.fixture(scope="session")
def device(gpus):
    new_device = torch.device("cpu")
    if gpus and use_cuda:
        device_id = gpus.split(",")
        assert len(device_id) == 2 and device_id[1] == ""
        device_id = int(gpus[0])
        new_device = torch.device(f"cuda:{device_id}")
        torch.cuda.set_device(new_device)

    return new_device


@pytest.fixture(scope="session")
def galaxy_decoder(data_path, device):
    dec_file = data_path.joinpath("galaxy_decoder_1_band.dat")
    dec = SimulatedDataset.get_gal_decoder_from_file(
        dec_file, gal_slen=51, n_bands=1, latent_dim=8
    )
    return dec


@pytest.fixture(scope="session")
def fitted_psf_params(data_path, device):
    psf_file = data_path.joinpath("fitted_powerlaw_psf_params.npy")
    psf_params = torch.from_numpy(np.load(psf_file)).to(device)
    return psf_params


@pytest.fixture(scope="session")
def get_star_dataset(device):
    def star_dataset(
        init_psf_params, batch_size=32, n_images=128, n_bands=1, slen=50, **dec_kwargs
    ):
        assert 1 <= n_bands <= 2

        dec_kwargs.update({"prob_galaxy": 0.0, "n_bands": n_bands, "slen": slen})
        background = torch.zeros(2, slen, slen, device=device)
        background[0] = 686.0
        background[1] = 1123.0

        # slice if necessary.
        background = background[range(n_bands)]
        init_psf_params = init_psf_params[None, 0, ...]

        dec_args = (None, init_psf_params, background)

        n_batches = int(n_images / batch_size)
        return SimulatedDataset(n_batches, batch_size, dec_args, dec_kwargs)

    return star_dataset


@pytest.fixture(scope="session")
def get_galaxy_dataset(device, galaxy_decoder, fitted_psf):
    def galaxy_dataset(batch_size=32, n_images=128, slen=10, **dec_kwargs):

        n_bands = 1

        # TODO: take background from test image.
        background = torch.zeros(n_bands, slen, slen, device=device)
        background[0] = 5000.0
        psf = fitted_psf[range(n_bands)]
        dec_args = (galaxy_decoder, psf, background)

        n_batches = int(n_images / batch_size)

        dec_kwargs.update({"prob_galaxy": 1.0, "n_bands": n_bands, "slen": slen})

        return SimulatedDataset(n_batches, batch_size, dec_args, dec_kwargs)

    return galaxy_dataset


@pytest.fixture(scope="session")
def get_trained_encoder(device, gpus):
    def trained_encoder(
        dataset,
        n_epochs=100,
        ptile_slen=8,
        step=2,
        edge_padding=3,
        enc_conv_c=5,
        enc_kern=3,
        enc_hidden=64,
        max_detections=2,
    ):
        n_epochs = n_epochs if use_cuda else 1

        slen = dataset.slen
        n_bands = dataset.n_bands
        latent_dim = dataset.image_decoder.latent_dim

        # setup Star Encoder
        encoder_kwargs = dict(
            ptile_slen=ptile_slen,
            step=step,
            edge_padding=edge_padding,
            enc_conv_c=enc_conv_c,
            enc_kern=enc_kern,
            enc_hidden=enc_hidden,
            max_detections=max_detections,
            slen=slen,
            n_bands=n_bands,
            n_galaxy_params=latent_dim,
        )

        sleep_net = sleep.SleepPhase(dataset, encoder_kwargs)

        sleep_trainer = pl.Trainer(
            gpus=gpus,
            min_epochs=n_epochs,
            max_epochs=n_epochs,
            reload_dataloaders_every_epoch=True,
            profiler=None,
            logger=False,
            checkpoint_callback=False,
        )

        sleep_trainer.fit(sleep_net)
        sleep_net.image_encoder.eval()
        return sleep_net.image_encoder

    return trained_encoder


@pytest.fixture(scope="session")
def test_3_stars(data_path):
    test_star = torch.load(data_path.joinpath("3_star_test.pt"))
    return test_star
