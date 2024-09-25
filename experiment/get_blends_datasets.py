#!/usr/bin/env python3

import datetime
from pathlib import Path

import click
import numpy as np
import pytorch_lightning as L
import torch

from bliss.datasets.generate_blends import generate_dataset
from bliss.datasets.lsst import (
    GALAXY_DENSITY,
    STAR_DENSITY,
    get_default_lsst_psf,
    prepare_final_galaxy_catalog,
    prepare_final_star_catalog,
)

HOME_DIR = Path(__file__).parent.parent.parent
CATSIM_CAT = prepare_final_galaxy_catalog()
STAR_MAGS = prepare_final_star_catalog()

PSF = get_default_lsst_psf()

TAG = datetime.datetime.now().strftime("%Y%m%d%H%M%S")


@click.command()
@click.option("-s", "--seed", required=True, type=int)
@click.option("-n", "--n-samples", default=1000, type=int)  # equally divided total blends
@click.option("--galaxy-density", default=GALAXY_DENSITY, type=float)
@click.option("--star-density", default=STAR_DENSITY, type=float)
def main(
    seed: int,
    n_samples: int,
    galaxy_density: float,
    star_density: float,
):

    L.seed_everything(seed)
    rng = np.random.default_rng(seed)  # for catalog indices

    train_ds_file = f"/nfs/turbo/lsa-regier/scratch/ismael/datasets/train_ds_{seed}_{TAG}.pt"
    val_ds_file = f"/nfs/turbo/lsa-regier/scratch/ismael/datasets/val_ds_{seed}_{TAG}.pt"
    test_ds_file = f"/nfs/turbo/lsa-regier/scratch/ismael/datasets/test_ds_{seed}_{TAG}.pt"

    if Path(train_ds_file).exists():
        raise IOError("Training file already exists")

    with open("run/log.txt", "a") as f:
        now = datetime.datetime.now()
        print("", file=f)
        log_msg = f"""Run training blend data generation script...
        With seed {seed} at {now}
        Galaxy density {galaxy_density}, star_density {star_density}, and n_samples {n_samples}.
        Samples will be divided into 3 datasets of blends with equal number.

        With TAG: {TAG}
        """
        print(log_msg, file=f)

    # disjointed tables with different galaxies
    n_rows = len(CATSIM_CAT)
    shuffled_indices = rng.choice(np.arange(n_rows), size=n_rows, replace=False)
    train_indices = shuffled_indices[: n_rows // 3]
    val_indices = shuffled_indices[n_rows // 3 : n_rows // 3 * 2]
    test_indices = shuffled_indices[n_rows // 3 * 2 : n_rows]
    table1 = CATSIM_CAT[train_indices]
    table2 = CATSIM_CAT[val_indices]
    table3 = CATSIM_CAT[test_indices]

    dss = []
    for t in (table1, table2, table3):
        dss.append(
            generate_dataset(
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
        )

    # now save data
    torch.save(dss[0], train_ds_file)
    torch.save(dss[1], val_ds_file)
    torch.save(dss[2], test_ds_file)


if __name__ == "__main__":
    main()
