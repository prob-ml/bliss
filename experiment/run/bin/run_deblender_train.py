#!/usr/bin/env python3


from pathlib import Path

import click

from bliss.encoders.deblend import GalaxyEncoder
from experiment.run.training_functions import run_encoder_training

NUM_WORKERS = 0


@click.command()
@click.option("-s", "--seed", required=True, type=int)
@click.option("--ds-seed", required=True, type=int)
@click.option("--ae-model-path", required=True, type=str)
@click.option("--train-file", required=True, type=str)
@click.option("--val-file", required=True, type=str)
@click.option("-b", "--batch-size", default=128)
@click.option("--lr", default=1e-4, type=float)
@click.option("-e", "--n-epochs", type=int, default=10_000)
@click.option("--validate-every-n-epoch", default=10, type=int)
@click.option("--log-every-n-steps", default=100, type=int)
def main(
    seed: int,
    ds_seed: int,
    ae_model_path: str,
    train_file: str,
    val_file: str,
    batch_size: int,
    lr: float,
    n_epochs: int,
    validate_every_n_epoch: int,
    log_every_n_steps: int,
):

    ae_path = Path(ae_model_path)
    assert ae_path.exists()

    # for logging
    info = {
        "ds_seed": ds_seed,
        "train_file": train_file,
        "val_file": val_file,
        "batch_size": batch_size,
        "n_epochs": n_epochs,
        "validate_every_n_epoch": validate_every_n_epoch,
        "val_check_interval": None,
        "lr": lr,
        "ae_path": ae_path,
    }

    # setup model to train
    galaxy_encoder = GalaxyEncoder(ae_path, lr=lr)

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
        keep_padding=True,
        log_info_dict=info,
    )


if __name__ == "__main__":
    main()
