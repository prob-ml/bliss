#!/usr/bin/env python3


import typer

from bliss.encoders.binary import BinaryEncoder
from bliss.training_functions import run_encoder_training

NUM_WORKERS = 0


def main(
    seed: int = typer.Option(),
    ds_seed: int = typer.Option(),
    train_file: str = typer.Option(),
    val_file: str = typer.Option(),
    batch_size: int = 32,
    n_epochs: int = 30,
    validate_every_n_epoch: int = 1,
    log_every_n_steps: int = 50,
    val_check_interval: float = 0.2,
):
    # for logging
    info = {
        "ds_seed": ds_seed,
        "train_file": train_file,
        "val_file": val_file,
        "batch_size": batch_size,
        "n_epochs": n_epochs,
        "validate_every_n_epoch": validate_every_n_epoch,
        "val_check_interval": val_check_interval,
        "lr": 1e-4,
    }

    binary_encoder = BinaryEncoder()

    run_encoder_training(
        seed=seed,
        train_file=train_file,
        val_file=val_file,
        batch_size=batch_size,
        n_epochs=n_epochs,
        model=binary_encoder,
        model_name="binary",
        validate_every_n_epoch=validate_every_n_epoch,
        val_check_interval=val_check_interval,
        log_every_n_steps=log_every_n_steps,
        log_info_dict=info,
    )


if __name__ == "__main__":
    typer.run(main)
