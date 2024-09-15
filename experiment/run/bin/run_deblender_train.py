#!/usr/bin/env python3


from pathlib import Path

import click

from bliss.encoders.deblend import GalaxyEncoder
from experiment.run.training_functions import run_encoder_training

NUM_WORKERS = 0
AE_STATE_DICT = "../models/autoencoder.pt"
assert Path(AE_STATE_DICT).exists()


@click.command()
@click.option("-s", "--seed", default=42, type=int)
@click.option("-b", "--batch-size", default=128)
@click.option("-e", "--n-epochs", default=10001)
@click.option("--validate-every-n-epoch", default=20, type=int)
@click.option("--log-every-n-steps", default=10, type=float, help="Fraction of training epoch")
@click.option("-t", "--tag", required=True, type=str, help="Dataset tag")
def main(
    seed: int,
    batch_size: int,
    n_epochs: int,
    validate_every_n_epoch: int,
    log_every_n_steps: int,
    tag: str,
):

    # setup model to train
    galaxy_encoder = GalaxyEncoder(AE_STATE_DICT)

    run_encoder_training(
        seed=seed,
        tag=tag,
        batch_size=batch_size,
        n_epochs=n_epochs,
        model=galaxy_encoder,
        model_name="deblender",
        validate_every_n_epoch=validate_every_n_epoch,
        val_check_interval=None,
        log_every_n_steps=log_every_n_steps,
    )


if __name__ == "__main__":
    main()
