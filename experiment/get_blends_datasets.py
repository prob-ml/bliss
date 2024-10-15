#!/usr/bin/env python3

import datetime
from pathlib import Path

import click
import numpy as np
import pytorch_lightning as L

from bliss import DATASETS_DIR, HOME_DIR
from bliss.datasets.generate_blends import generate_dataset
from bliss.datasets.io import save_dataset_npz
from bliss.datasets.lsst import (
    GALAXY_DENSITY,
    STAR_DENSITY,
    get_default_lsst_psf,
    prepare_final_galaxy_catalog,
    prepare_final_star_catalog,
)

LOG_FILE = HOME_DIR / "experiment/log.txt"

CATSIM_CAT = prepare_final_galaxy_catalog()
STAR_MAGS = prepare_final_star_catalog()

PSF = get_default_lsst_psf()


assert LOG_FILE.exists()


@click.command()
@click.option("-s", "--seed", required=True, type=int)
@click.option("-n", "--n-samples", default=50_000, type=int)  # equally divided total blends
@click.option("--galaxy-density", default=GALAXY_DENSITY, type=float)
@click.option("--star-density", default=STAR_DENSITY, type=float)
def main(seed: int, n_samples: int, galaxy_density: float, star_density: float):

    L.seed_everything(seed)
    rng = np.random.default_rng(seed)  # for catalog indices

    train_ds_file = DATASETS_DIR / f"train_ds_{seed}.npz"
    val_ds_file = DATASETS_DIR / f"val_ds_{seed}.npz"
    test_ds_file = DATASETS_DIR / f"test_ds_{seed}.npz"

    assert not train_ds_file.exists(), "files exist"
    assert not val_ds_file.exists(), "files exist"
    assert not test_ds_file.exists(), "files exist"

    # disjointed tables with different galaxies
    n_rows = len(CATSIM_CAT)
    shuffled_indices = rng.choice(np.arange(n_rows), size=n_rows, replace=False)
    train_indices = shuffled_indices[: n_rows // 3]
    val_indices = shuffled_indices[n_rows // 3 : n_rows // 3 * 2]
    test_indices = shuffled_indices[n_rows // 3 * 2 :]

    table1 = CATSIM_CAT[train_indices]
    table2 = CATSIM_CAT[val_indices]
    table3 = CATSIM_CAT[test_indices]

    files = (train_ds_file, val_ds_file, test_ds_file)
    tables = (table1, table2, table3)
    for f, t in zip(files, tables):
        ds = generate_dataset(
            n_samples,
            t,
            STAR_MAGS,
            psf=PSF,
            max_n_sources=10,
            galaxy_density=galaxy_density,
            star_density=star_density,
            slen=40,
            bp=24,
            max_shift=0.5,
        )
        save_dataset_npz(ds, f)

    # logging
    with open(LOG_FILE, "a") as f:
        now = datetime.datetime.now()
        log_msg = f"""\nBlend data generation with seed {seed} at {now}.
        Galaxy density {galaxy_density}, star_density {star_density}, and n_samples {n_samples}.
        """
        print(log_msg, file=f)


if __name__ == "__main__":
    main()
