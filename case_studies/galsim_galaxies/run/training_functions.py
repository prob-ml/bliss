import sys
from typing import TextIO

import pytorch_lightning as L
import torch
from astropy.table import Table
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

from bliss.datasets.galsim_blends import SavedGalsimBlends, generate_dataset
from bliss.datasets.lsst import get_default_lsst_psf
from bliss.datasets.table_utils import column_to_tensor


def create_dataset(
    catsim_file: str,
    stars_mag_file: str,
    n_samples: int,
    train_val_split: int,
    train_ds_file: str,
    val_ds_file: str,
    only_bright=False,
    add_galaxies_in_padding=True,
    galaxy_density: float = 185,
    star_density: float = 10,
    log_file: TextIO = sys.stdout,
):
    print("INFO: Overwriting dataset...", file=log_file)

    # prepare bigger dataset
    catsim_table = Table.read(catsim_file)
    all_star_mags = column_to_tensor(Table.read(stars_mag_file), "i_ab")
    psf = get_default_lsst_psf()

    if only_bright:
        bright_mask = catsim_table["i_ab"] < 23
        new_table = catsim_table[bright_mask]
        print(
            "INFO: Smaller catalog with only bright sources of length:",
            len(new_table),
            file=log_file,
        )

    else:
        mask = catsim_table["i_ab"] < 27.3
        new_table = catsim_table[mask]
        print(
            "INFO: Complete catalog with only i < 27.3 magnitude of length:",
            len(new_table),
            file=log_file,
        )

    dataset = generate_dataset(
        n_samples,
        new_table,
        all_star_mags,
        psf=psf,
        max_n_sources=15,  # https://www.wolframalpha.com/input?i=Poisson+distribution+with+mean+4
        slen=40,
        bp=24,
        max_shift=0.5,
        add_galaxies_in_padding=add_galaxies_in_padding,
        galaxy_density=galaxy_density,
        star_density=star_density,
    )

    # train, test split
    train_ds = {p: q[:train_val_split] for p, q in dataset.items()}
    val_ds = {p: q[train_val_split:] for p, q in dataset.items()}

    # now save  data
    torch.save(train_ds, train_ds_file)
    torch.save(val_ds, val_ds_file)


def setup_training_objects(
    train_ds_file: str,
    val_ds_file: str,
    n_samples: int,
    train_val_split: int,
    batch_size: int,
    num_workers: int,
    n_epochs: int,
    validate_every_n_epoch: int,
    val_check_interval: float,
    log_file: TextIO = sys.stdout,
):
    train_dataset = SavedGalsimBlends(train_ds_file, train_val_split)
    validation_dataset = SavedGalsimBlends(val_ds_file, n_samples - train_val_split)
    train_dl = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_dl = DataLoader(validation_dataset, batch_size=batch_size, num_workers=num_workers)

    ccb = ModelCheckpoint(
        filename="epoch={epoch}-val_loss={val/loss:.3f}",
        save_top_k=5,
        verbose=True,
        monitor="val/loss",
        mode="min",
        save_on_train_epoch_end=False,
        auto_insert_metric_name=False,
    )

    logger = TensorBoardLogger(save_dir="out", name="detection", default_hp_metric=False)
    print(f"INFO: Saving model as version {logger.version}", file=log_file)

    trainer = L.Trainer(
        limit_train_batches=1.0,
        max_epochs=n_epochs,
        logger=logger,
        callbacks=[ccb],
        accelerator="gpu",
        devices=1,
        log_every_n_steps=16,
        check_val_every_n_epoch=validate_every_n_epoch,
        val_check_interval=val_check_interval,
    )

    return train_dl, val_dl, trainer
