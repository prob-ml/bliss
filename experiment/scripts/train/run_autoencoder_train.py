#!/usr/bin/env python3

from pathlib import Path

import pytorch_lightning as L
import typer

from bliss.datasets.saved_datasets import SavedIndividualGalaxies
from bliss.encoders.autoencoder import OneCenteredGalaxyAE
from bliss.training_functions import setup_training_objects
from experiment import MODELS_DIR

NUM_WORKERS = 0


def main(
    seed: int = typer.Option(),
    train_file: str = typer.Option(),
    val_file: str = typer.Option(),
    batch_size: int = 128,
    n_epochs: int = 10_000,
    validate_every_n_epoch: int = 10,
    lr: float = 1e-5,
    version: int = 0,
):
    L.seed_everything(seed)

    assert not (MODELS_DIR / f"autoencoder_{seed}.pt").exists(), "model exists."
    assert Path(train_file).exists(), f"Training dataset {train_file} is not available"
    assert Path(val_file).exists(), f"Training dataset {val_file} is not available"

    # setup model to train
    autoencoder = OneCenteredGalaxyAE(lr=lr)

    train_ds = SavedIndividualGalaxies(train_file)
    val_ds = SavedIndividualGalaxies(val_file)
    train_dl, val_dl, trainer, _ = setup_training_objects(
        train_ds,
        val_ds,
        batch_size,
        NUM_WORKERS,
        n_epochs,
        validate_every_n_epoch=validate_every_n_epoch,
        val_check_interval=None,
        model_name="autoencoder",
        log_every_n_steps=train_ds.epoch_size // batch_size,  # = number of batches in 1 epoch
        version=version,
    )

    # fit!
    trainer.fit(model=autoencoder, train_dataloaders=train_dl, val_dataloaders=val_dl)


if __name__ == "__main__":
    typer.run(main)
