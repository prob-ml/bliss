#!/usr/bin/env python3

import datetime
from pathlib import Path

import click
import pytorch_lightning as L

from bliss.datasets.galsim_blends import SavedIndividualGalaxies
from bliss.encoders.autoencoder import OneCenteredGalaxyAE
from experiment.run.training_functions import setup_training_objects

NUM_WORKERS = 0


@click.command()
@click.option("-s", "--seed", default=42, type=int)
@click.option("-b", "--batch-size", default=128)
@click.option("-e", "--n-epochs", default=10001)
@click.option("--validate-every-n-epoch", default=1, type=int)
@click.option("-t", "--tag", required=True, type=str, help="Dataset tag")
@click.option("--lr", default=1e-4, type=float)
def main(
    seed: int,
    batch_size: int,
    n_epochs: int,
    validate_every_n_epoch: int,
    tag: str,
    lr: float,
):

    with open("log.txt", "a") as f:
        now = datetime.datetime.now()
        print("", file=f)
        log_msg = f"""Run training autoencoder script...
        With tag {tag} and seed {seed} at {now}
        validate_every_n_epoch {validate_every_n_epoch},
        batch_size {batch_size}, n_epochs {n_epochs}
        learning rate {lr}
        """
        print(log_msg, file=f)

    L.seed_everything(seed)

    train_ds_file = f"/nfs/turbo/lsa-regier/scratch/ismael/datasets/train_ae_ds_{tag}.pt"
    val_ds_file = f"/nfs/turbo/lsa-regier/scratch/ismael/datasets/val_ae_ds_{tag}.pt"

    assert Path(train_ds_file).exists(), f"Training dataset with tag {tag} is not available"

    # setup model to train
    autoencoder = OneCenteredGalaxyAE(lr=lr)

    with open("log.txt", "a") as g:
        train_ds = SavedIndividualGalaxies(train_ds_file)
        val_ds = SavedIndividualGalaxies(val_ds_file)
        train_dl, val_dl, trainer = setup_training_objects(
            train_ds,
            val_ds,
            batch_size,
            NUM_WORKERS,
            n_epochs,
            validate_every_n_epoch=validate_every_n_epoch,
            val_check_interval=None,
            model_name="autoencoder",
            log_every_n_steps=train_ds.epoch_size // batch_size,  # = number of batches in 1 epoch
            log_file=g,
        )

    trainer.fit(model=autoencoder, train_dataloaders=train_dl, val_dataloaders=val_dl)


if __name__ == "__main__":
    main()
