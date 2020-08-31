import pytest
from pytorch_lightning import Callback
import optuna
from optuna.trial import FixedTrial

from bliss.sleep import SleepObjective


class MetricsCallback(Callback):
    """PyTorch Lightning metric callback."""

    def __init__(self):
        super().__init__()
        self.metrics = []

    def on_validation_end(self, trainer, pl_module):
        self.metrics.append(trainer.callback_metrics)


@pytest.fixture(scope="module")
def metrics_callback_setup(device_setup):
    return MetricsCallback()


@pytest.fixture(scope="module")
def star_dataset(decoder_setup, device_setup):
    psf_params = decoder_setup.get_fitted_psf_params()
    batch_size = 32 if device_setup.use_cuda else 1
    n_images = 128 if device_setup.use_cuda else 1
    return decoder_setup.get_star_dataset(
        psf_params, n_bands=1, slen=50, batch_size=batch_size, n_images=n_images
    )


class TestOptunaSleep:
    def test_optuna(self, star_dataset, metrics_callback_setup, paths, device_setup):
        # set up encoder
        encoder_kwargs = dict(
            enc_conv_c=(5, 25, 5),
            enc_kern=3,
            enc_hidden=(64, 128, 64),
            ptile_slen=8,
            max_detections=2,
            slen=star_dataset.slen,
            n_bands=star_dataset.n_bands,
            n_galaxy_params=star_dataset.latent_dim,
        )

        n_epochs = 100 if device_setup.use_cuda else 1
        # set up Object for optuna
        objects = SleepObjective(
            star_dataset,
            encoder_kwargs,
            max_epochs=n_epochs,
            lr=(1e-4, 1e-2),
            weight_decay=(1e-6, 1e-4),
            model_dir=paths["model_dir"],
            metrics_callback=metrics_callback_setup,
            monitor="val_loss",
            gpus=device_setup.gpus,
        )

        # set up study object
        objects(
            FixedTrial(
                {
                    "enc_conv_c": 5,
                    "enc_hidden": 64,
                    "learning rate": 1e-3,
                    "weight_decay": 1e-5,
                }
            )
        )
