#!/usr/bin/env python3
"""We generate 3 different datasets for each type of encoder."""

import datetime

import numpy as np
import pytorch_lightning as L
import typer

from bliss import DATASETS_DIR, HOME_DIR
from bliss.datasets.io import save_dataset_npz
from bliss.datasets.lsst import (
    GALAXY_DENSITY,
    STAR_DENSITY,
    get_default_lsst_psf,
    prepare_final_galaxy_catalog,
    prepare_final_star_catalog,
)
from bliss.datasets.padded_tiles import generate_padded_tiles

LOG_FILE = HOME_DIR / "experiment/log.txt"

CATSIM_CAT = prepare_final_galaxy_catalog()
STAR_MAGS = prepare_final_star_catalog()

PSF = get_default_lsst_psf()


assert LOG_FILE.exists()


def main(
    seed: int = typer.Option(),
    indices_fname: str = typer.Option(),
    n_train: int = 50000,
    n_val: int = 10000,
    galaxy_density: float = GALAXY_DENSITY,
    star_density: float = STAR_DENSITY,
):
    L.seed_everything(seed)

    # disjointed tables with different galaxies
    assert indices_fname.endswith(".npz")
    indices_fpath = DATASETS_DIR / indices_fname
    assert indices_fpath.exists()
    indices_dict = np.load(indices_fpath)
    train_indices = indices_dict["train"]
    val_indices = indices_dict["val"]

    # galxies, centered, no empty tiles
    train_ds_deblend_file = DATASETS_DIR / f"train_ds_deblend_{seed}.npz"
    val_ds_deblend_file = DATASETS_DIR / f"val_ds_deblend_{seed}.npz"

    table1 = CATSIM_CAT[train_indices]
    table2 = CATSIM_CAT[val_indices]

    # deblend
    assert not train_ds_deblend_file.exists(), "files exist"
    assert not val_ds_deblend_file.exists(), "files exist"

    ds1 = generate_padded_tiles(
        n_train,
        table1,
        STAR_MAGS,
        psf=PSF,
        max_shift=0.5,
        p_source_in=1.0,
        galaxy_prob=1.0,
    )
    ds2 = generate_padded_tiles(
        n_val,
        table2,
        STAR_MAGS,
        psf=PSF,
        max_shift=0.5,
        p_source_in=1.0,
        galaxy_prob=1.0,
    )
    save_dataset_npz(ds1, train_ds_deblend_file)
    save_dataset_npz(ds2, val_ds_deblend_file)

    # logging
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        now = datetime.datetime.now()
        log_msg = f"""Tile test data generation with seed {seed} at {now}.
        Galaxy density {galaxy_density}, star_density {star_density}, and n_samples {n_train}.
        """
        print(log_msg, file=f)


if __name__ == "__main__":
    typer.run(main)
