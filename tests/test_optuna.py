import pytest
import os
import torch
import pytorch_lightning as pl
from pytorch_lightning import Callback
import optuna
from optuna.trial import FixedTrial
from optuna.integration import PyTorchLightningPruningCallback


DIR = os.getcwd()
MODEL_DIR = os.path.join(DIR, "result")


class MetricsCallback(Callback):
    """PyTorch Lightning metric callback."""

    def __init__(self):
        super().__init__()
        self.metrics = []

    def on_validation_end(self, trainer, pl_module):
        self.metrics.append(trainer.callback_metrics)


@pytest.fixture(scope="module")
def star_dataset(decoder_setup):
    psf_params = decoder_setup.get_fitted_psf_params()
    return decoder_setup.get_star_dataset(psf_params, n_bands=1, slen=50, batch_size=32)


class Objective(object):
    def __init__(self, star_dataset, encoder_setup, device_setup):
        self.n_epochs = 100 if device_setup.use_cuda else 1
        self.star_dataset = star_dataset
        self.encoder_setup = encoder_setup

    def __call__(self, trial):
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            os.path.join(MODEL_DIR, "trial_{}".format(trial.number), "{epoch}"),
            monitor="val_acc",
        )
        enc_conv_c = trial.suggest_int("enc_conv_c", 5, 25, 5)
        enc_kern = trial.suggest_int("enc_kern", 3, 6, 1)
        enc_hidden = trial.suggest_int("enc_hidden", 64, 256, 64)
        trained_encoder = self.encoder_setup.get_trained_encoder(
            star_dataset, enc_conv_c, enc_kern, enc_hidden, n_epochs=self.n_epochs
        )

        return trained_encoder


def test_optuna(star_dataset, encoder_setup, device_setup):
    study = optuna.create_study()
    study.optimize(
        Objective(
            star_dataset,
            encoder_setup,
            device_setup,
            FixedTrial({"enc_conv_c": 5, "enc_kern": 3, "enc_hidden": 64}),
        )
    )
