#!/usr/bin/env python3

import datetime
from pathlib import Path

import click
import pytorch_lightning as L

from bliss import HOME_DIR
from bliss.datasets.saved_datasets import SavedIndividualGalaxies
from bliss.encoders.autoencoder import OneCenteredGalaxyAE
from bliss.training_functions import setup_training_objects

NUM_WORKERS = 0

MODELS_DIR = HOME_DIR / "experiment/models"
LOG_FILE = HOME_DIR / "experiment/log.txt"
LOG_FILE_LONG = HOME_DIR / "experiment/log_long.txt"


def _log_info(seed, info: dict):
    now = datetime.datetime.now()

    ds_seed = info["ds_seed"]
    validate_every_n_epoch = info["validate_every_n_epoch"]
    batch_size = info["batch_size"]
    n_epochs = info["n_epochs"]
    lr = info["lr"]
    train_file = info["train_file"]
    val_file = info["val_file"]
    vnum = info["version"]

    log_msg_short = f"\nTraining AE with seed {seed}, ds_seed {ds_seed}, version {vnum} at {now}."
    log_msg_long = f"""{log_msg_short}

    validate_every_n_epoch {validate_every_n_epoch},
    batch_size {batch_size}, n_epochs {n_epochs}
    learning rate {lr}

    Using datasets: {train_file}, {val_file}
    """
    with open(LOG_FILE, "a", encoding="utf-8") as f1:
        print(log_msg_short, file=f1)

    with open(LOG_FILE_LONG, "a", encoding="utf-8") as f2:
        print("", file=f2)
        print(log_msg_long, file=f2)


@click.command()
@click.option("-s", "--seed", required=True, type=int)
@click.option("--ds-seed", required=True, type=int, help="Random seed used for dataset")
@click.option("--train-file", required=True, type=str)
@click.option("--val-file", required=True, type=str)
@click.option("-b", "--batch-size", default=128)
@click.option("-e", "--n-epochs", default=10_000)
@click.option("--validate-every-n-epoch", default=10, type=int)
@click.option("--lr", default=1e-5, type=float)
@click.option("--version", default="", type=str)
def main(
    seed: int,
    ds_seed: int,
    train_file: str,
    val_file: str,
    batch_size: int,
    n_epochs: int,
    validate_every_n_epoch: int,
    lr: float,
    version: str,
):
    # setup version
    if not version:
        version = None
    else:
        try:
            version = int(version)
        except ValueError:
            raise ValueError("Version must be a number if passed in.")

    L.seed_everything(seed)

    assert not (MODELS_DIR / f"autoencoder_{ds_seed}_{seed}.pt").exists(), "model exists."
    assert Path(train_file).exists(), f"Training dataset {train_file} is not available"
    assert Path(val_file).exists(), f"Training dataset {val_file} is not available"

    # setup model to train
    autoencoder = OneCenteredGalaxyAE(lr=lr)

    train_ds = SavedIndividualGalaxies(train_file)
    val_ds = SavedIndividualGalaxies(val_file)
    train_dl, val_dl, trainer, vnum = setup_training_objects(
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

    # logging
    info = {
        "ds_seed": ds_seed,
        "train_file": train_file,
        "val_file": val_file,
        "batch_size": batch_size,
        "n_epochs": n_epochs,
        "validate_every_n_epoch": validate_every_n_epoch,
        "lr": lr,
        "version": vnum,
    }
    _log_info(seed, info)

    # fit!
    trainer.fit(model=autoencoder, train_dataloaders=train_dl, val_dataloaders=val_dl)


if __name__ == "__main__":
    main()
