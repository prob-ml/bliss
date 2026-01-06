#!/usr/bin/env python3


import numpy as np
import pytorch_lightning as L
import typer

from bliss.datasets.generate_blends import generate_dataset
from bliss.datasets.io import save_dataset_npz
from bliss.datasets.lsst import (
    GALAXY_DENSITY,
    STAR_DENSITY,
    get_default_lsst_psf,
    prepare_final_galaxy_catalog,
    prepare_final_star_catalog,
)
from experiment import DATASETS_DIR

CATSIM_CAT = prepare_final_galaxy_catalog()
STAR_MAGS = prepare_final_star_catalog()

PSF = get_default_lsst_psf()


def main(
    seed: int = typer.Option(),
    indices_fname: str = typer.Option(),
    n_samples: int = 10000,
    galaxy_density: float = GALAXY_DENSITY,
    star_density: float = STAR_DENSITY,
):
    L.seed_everything(seed)

    train_ds_file = DATASETS_DIR / f"train_ds_{seed}.npz"
    val_ds_file = DATASETS_DIR / f"val_ds_{seed}.npz"
    test_ds_file = DATASETS_DIR / f"test_ds_{seed}.npz"

    assert not train_ds_file.exists(), "files exist"
    assert not val_ds_file.exists(), "files exist"
    assert not test_ds_file.exists(), "files exist"

    # disjointed tables with different galaxies
    indices_fpath = DATASETS_DIR / indices_fname
    assert indices_fpath.exists(), "indices file does not exist."
    indices_dict = np.load(indices_fpath)
    train_indices = indices_dict["train"]
    val_indices = indices_dict["val"]
    test_indices = indices_dict["test"]

    table1 = CATSIM_CAT[train_indices]
    table2 = CATSIM_CAT[val_indices]
    table3 = CATSIM_CAT[test_indices]

    files = (train_ds_file, val_ds_file, test_ds_file)
    tables = (table1, table2, table3)
    _samples = (n_samples, int(n_samples * 0.5), int(n_samples * 1.5))
    for fpath, t, ns in zip(files, tables, _samples, strict=True):
        ds = generate_dataset(
            ns,
            t,
            STAR_MAGS,
            psf=PSF,
            max_n_sources=10,
            galaxy_density=galaxy_density,
            star_density=star_density,
            max_shift=0.5,
        )
        save_dataset_npz(ds, fpath)


if __name__ == "__main__":
    typer.run(main)
