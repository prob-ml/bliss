#!/usr/bin/env python3


from pathlib import Path

import click
from pytorch_lightning.callbacks import EarlyStopping

from bliss.encoders.deblend import GalaxyEncoder
from experiment.run.training_functions import run_encoder_training

NUM_WORKERS = 0
AE_STATE_DICT = "../models/autoencoder.pt"
assert Path(AE_STATE_DICT).exists()


@click.command()
@click.option("-s", "--seed", required=True, type=int)
@click.option("--train-file", required=True, type=str)
@click.option("--val-file", required=True, type=str)
@click.option("-b", "--batch-size", default=128)
@click.option("-e", "--n-epochs", default=10001)
@click.option("--validate-every-n-epoch", default=20, type=int)
@click.option("--log-every-n-steps", default=10, type=float, help="Fraction of training epoch")
def main(
    seed: int,
    train_file: str,
    val_file: str,
    batch_size: int,
    n_epochs: int,
    validate_every_n_epoch: int,
    log_every_n_steps: int,
):

    # setup model to train
    galaxy_encoder = GalaxyEncoder(AE_STATE_DICT)

    # early stoppin callback based on 'mean_max_residual'
    early_stopping_cb = EarlyStopping(
        "val/mean_max_residual",
        min_delta=0.1,
        patience=10,
        strict=True,
        check_on_train_epoch_end=False,
        mode="min",
    )

    run_encoder_training(
        seed=seed,
        train_file=train_file,
        val_file=val_file,
        batch_size=batch_size,
        n_epochs=n_epochs,
        model=galaxy_encoder,
        model_name="deblender",
        validate_every_n_epoch=validate_every_n_epoch,
        val_check_interval=None,
        log_every_n_steps=log_every_n_steps,
        early_stopping_cb=early_stopping_cb,
    )


if __name__ == "__main__":
    main()
