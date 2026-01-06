#!/usr/bin/env python3
"""We generate 3 different datasets for each type of encoder."""

import numpy as np
import pytorch_lightning as L
import typer

from bliss.datasets.io import save_dataset_npz
from bliss.datasets.lsst import (
    get_default_lsst_psf,
    prepare_final_galaxy_catalog,
    prepare_final_star_catalog,
)
from bliss.datasets.padded_tiles import generate_padded_tiles
from experiment import DATASETS_DIR

CATSIM_CAT = prepare_final_galaxy_catalog()
STAR_MAGS = prepare_final_star_catalog()

PSF = get_default_lsst_psf()


def main(
    seed: int = typer.Option(),
    indices_fname: str = typer.Option(),
    n_train: int = 50000,
    n_val: int = 10000,
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


if __name__ == "__main__":
    typer.run(main)
