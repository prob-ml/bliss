#!/usr/bin/env python3

from pathlib import Path

import click
import torch
from astropy.table import Table

from bliss.datasets.galsim_blends import generate_dataset
from bliss.datasets.lsst import get_default_lsst_psf
from bliss.datasets.table_utils import column_to_tensor

HOME_DIR = Path(__file__).parent.parent.parent
DATA_DIR = Path(__file__).parent / "data"
CATSIM_TABLE = Table.read(HOME_DIR / "data/OneDegSq.fits")
STAR_MAGS = column_to_tensor(Table.read(HOME_DIR / "data/stars_med_june2018.fits"), "i_ab")
PSF = get_default_lsst_psf()


@click.command()
@click.option("--n-samples", default=3000)
def main(n_samples: int):
    assert n_samples % 3 == 0
    dataset = generate_dataset(
        n_samples,
        CATSIM_TABLE,
        STAR_MAGS,
        mean_sources=5,
        max_n_sources=10,
        psf=PSF,
        slen=40,
        bp=24,
        max_shift=0.5,
        galaxy_prob=0.95,
    )

    # split into three
    div1, div2 = n_samples // 3, n_samples // 3 * 2
    train_ds = {p: q[:div1] for p, q in dataset.items()}
    val_ds = {p: q[div1:div2] for p, q in dataset.items()}
    test_ds = {p: q[div2:] for p, q in dataset.items()}

    torch.save(train_ds, DATA_DIR / "train_gbs.pt")
    torch.save(val_ds, DATA_DIR / "val_gbs.pt")
    torch.save(test_ds, DATA_DIR / "test_gbs.pt")


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
