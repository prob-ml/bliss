import pytest
import torch
from pytorch_lightning import Callback
from optuna.trial import FixedTrial

from bliss.hyperparameter import SleepObjective


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


class TestOptunaSleep:
    def test_optuna(self, decoder_setup, metrics_callback_setup, paths, device_setup):

        # psf
        psf_params = decoder_setup.get_fitted_psf_params()

        # background
        background = torch.zeros(1, 50, 50)
        background[0] = 686.0

        # decoder arguments
        dec_args = (None, psf_params, background)
        dec_kwargs = {}
        dec_kwargs.update({"prob_galaxy": 0.0, "n_bands": 1, "slen": 50})

        # set up encoder
        encoder_kwargs = dict(
            enc_conv_c=(5, 25, 5),
            enc_kern=3,
            enc_hidden=(64, 128, 64),
            ptile_slen=8,
            max_detections=2,
            slen=50,
            n_bands=1,
            n_galaxy_params=8,
        )

        n_epochs = 1
        # set up Object for optuna
        objects = SleepObjective(
            encoder_kwargs,
            max_epochs=n_epochs,
            lr=(1e-4, 1e-2),
            weight_decay=(1e-6, 1e-4),
            model_dir=paths["model_dir"],
            metrics_callback=metrics_callback_setup,
            monitor="val_loss",
            n_batches=4,
            batch_size=32,
            dec_args=dec_args,
            dec_kwargs=dec_kwargs,
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
