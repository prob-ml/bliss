#!/usr/bin/env python3


from pathlib import Path

import typer

from bliss.encoders.deblend import GalaxyEncoder
from bliss.training_functions import run_encoder_training

NUM_WORKERS = 0


def main(
    seed: int = typer.Option(),
    ds_seed: int = typer.Option(),
    ae_model_path: str = typer.Option(),
    train_file: str = typer.Option(),
    val_file: str = typer.Option(),
    batch_size: int = 128,
    lr: float = 1e-4,
    n_epochs: int = 10_000,
    validate_every_n_epoch: int = 10,
    log_every_n_steps: int = 25,
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
        is_deblender=True,
        log_info_dict=info,
    )


if __name__ == "__main__":
    typer.run(main)
