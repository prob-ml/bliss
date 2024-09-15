#!/usr/bin/env python3

import datetime
from pathlib import Path

import click
import numpy as np
import pytorch_lightning as L
import torch
from astropy.table import Table

from bliss.datasets.galsim_blends import generate_dataset
from bliss.datasets.lsst import get_default_lsst_psf
from bliss.datasets.table_utils import column_to_tensor

HOME_DIR = Path(__file__).parent.parent.parent
CATSIM_TABLE = Table.read(HOME_DIR / "data" / "catsim_snr.fits")
_stars_mags = column_to_tensor(Table.read(HOME_DIR / "data" / "stars_med_june2018.fits"), "i_ab")


# we mask out stars with mag < 20 which corresponds to SNR >1000
# as the notebook `test-stars-with-new-model` shows.
STAR_MAGS = _stars_mags[_stars_mags > 20]  # remove very bright stars


PSF = get_default_lsst_psf()
assert np.all(CATSIM_TABLE["i_ab"].value < 27.3)


@click.command()
@click.option("-s", "--seed", default=42, type=int)
@click.option("-t", "--tag", required=True, type=str, help="Dataset tag")
@click.option("-n", "--n-samples", default=10000, type=int)  # equally divided total blends
@click.option("--only-bright", is_flag=True, default=False)
@click.option("--no-padding-galaxies", is_flag=True, default=False)
@click.option("--galaxy-density", default=185, type=float)
@click.option("--star-density", default=10, type=float)
def main(
    seed: int,
    tag: str,
    n_samples: int,
    only_bright: bool,
    no_padding_galaxies: bool,
    galaxy_density: float,
    star_density: float,
):

    L.seed_everything(seed)

    train_ds_file = f"/nfs/turbo/lsa-regier/scratch/ismael/datasets/train_ds_{tag}.pt"
    val_ds_file = f"/nfs/turbo/lsa-regier/scratch/ismael/datasets/val_ds_{tag}.pt"
    test_ds_file = f"/nfs/turbo/lsa-regier/scratch/ismael/datasets/test_ds_{tag}.pt"

    if Path(train_ds_file).exists():
        raise IOError("Training file already exists")

    with open("run/log.txt", "a") as f:
        now = datetime.datetime.now()
        print("", file=f)
        log_msg = f"""Run training blend data generation script...
        With tag {tag} and seed {seed} at {now}
        Galaxy density {galaxy_density}, star_density {star_density}, and
        Only bright '{only_bright}' (defined with snr > 10),
        no padding galaxies '{no_padding_galaxies}', n_samples {n_samples}.
        Samples will be divided into 3 datasets of blends with equal number.
        """
        print(log_msg, file=f)

        if only_bright:
            bright_mask = CATSIM_TABLE["snr"] > 10
            new_table = CATSIM_TABLE[bright_mask]
            print(
                "INFO: Smaller catalog with only bright sources of length:",
                len(new_table),
                file=f,
            )
        else:
            new_table = CATSIM_TABLE

        print(
            "INFO: Removing bright stars with i < 20 magnitude, final catalog length:",
            len(STAR_MAGS),
            file=f,
        )

    n_rows = len(new_table)
    shuffled_indices = np.random.choice(np.arange(n_rows), size=n_rows, replace=False)
    train_indices = shuffled_indices[: n_rows // 3]
    val_indices = shuffled_indices[n_rows // 3 : n_rows // 3 * 2]
    test_indices = shuffled_indices[n_rows // 3 * 2 : n_rows]
    table1 = new_table[train_indices]
    table2 = new_table[val_indices]
    table3 = new_table[test_indices]

    dss = []
    for t in (table1, table2, table3):
        dss.append(
            generate_dataset(
                n_samples,
                t,
                STAR_MAGS,
                psf=PSF,
                # https://www.wolframalpha.com/input?i=poisson+distribution+with+mean+3.5
                max_n_sources=10,
                galaxy_density=galaxy_density,
                star_density=star_density,
                slen=40,
                bp=24,
                max_shift=0.5,
                add_galaxies_in_padding=not no_padding_galaxies,
            )
        )

    # now save data
    torch.save(dss[0], train_ds_file)
    torch.save(dss[1], val_ds_file)
    torch.save(dss[2], test_ds_file)


if __name__ == "__main__":
    main()
