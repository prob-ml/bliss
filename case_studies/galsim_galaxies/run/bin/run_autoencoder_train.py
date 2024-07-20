#!/usr/bin/env python3

import datetime
from pathlib import Path

import click
import pytorch_lightning as L

from bliss.datasets.galsim_blends import SavedIndividualGalaxies
from bliss.encoders.autoencoder import OneCenteredGalaxyAE
from case_studies.galsim_galaxies.run.training_functions import (
    create_dataset,
    setup_training_objects,
)

NUM_WORKERS = 0


@click.command()
@click.option("-s", "--seed", default=42, type=int)
@click.option("-n", "--n-samples", default=1028 * 100, type=int)
@click.option("--split", default=1028 * 75, type=int)
@click.option("-b", "--batch-size", default=256)
@click.option("-e", "--n-epochs", default=3001)
@click.option("--validate-every-n-epoch", default=1, type=int)
@click.option("-o", "--overwrite", is_flag=True, default=False)
@click.option("-t", "--tag", required=True, type=str, help="Dataset tag")
@click.option("--only-bright", is_flag=True, default=False)
def main(
    seed: int,
    n_samples: int,
    split: int,
    batch_size: int,
    n_epochs: int,
    validate_every_n_epoch: int,
    overwrite: bool,
    tag: str,
    only_bright: bool,
):

    with open("log.txt", "a") as f:
        now = datetime.datetime.now()
        print("", file=f)
        log_msg = f"""Run training autoencoder script...
        With tag {tag} and seed {seed} at {now}
        Only bright '{only_bright}',
        n_samples {n_samples}, split {split}, validate_every_n_epoch {validate_every_n_epoch},
        batch_size {batch_size}, n_epochs {n_epochs}
        """
        print(log_msg, file=f)

    L.seed_everything(seed)

    train_ds_file = f"/nfs/turbo/lsa-regier/ismael/datasets/train_ae_ds_{tag}.pt"
    val_ds_file = f"/nfs/turbo/lsa-regier/ismael/datasets/val_ae_ds_{tag}.pt"

    # setup model to train
    autoencoder = OneCenteredGalaxyAE()

    if overwrite:
        with open("log.txt", "a") as f:
            create_dataset(
                catsim_file="../../../data/OneDegSq.fits",
                stars_mag_file="../../../data/stars_med_june2018.fits",  # will not be used
                n_samples=n_samples,
                train_val_split=split,
                train_ds_file=train_ds_file,
                val_ds_file=val_ds_file,
                only_bright=only_bright,
                add_galaxies_in_padding=False,
                galaxy_density=1000,  # hack to always have at least 1 galaxy
                star_density=0,
                max_n_sources=1,
                slen=53,
                bp=0,
                max_shift=0,  # centered
                log_file=f,
            )

    if not overwrite and not Path(val_ds_file).exists():
        raise IOError("Validation dataset file not found and overwrite is 'False'.")

    with open("log.txt", "a") as g:
        train_ds = SavedIndividualGalaxies(train_ds_file, split)
        val_ds = SavedIndividualGalaxies(val_ds_file, n_samples - split)
        train_dl, val_dl, trainer = setup_training_objects(
            train_ds,
            val_ds,
            batch_size,
            NUM_WORKERS,
            n_epochs,
            validate_every_n_epoch,
            val_check_interval=None,
            model_name="autoencoder",
            log_every_n_steps=200,
            log_file=g,
        )

    trainer.fit(model=autoencoder, train_dataloaders=train_dl, val_dataloaders=val_dl)


if __name__ == "__main__":
    main()
