#!/usr/bin/env python3

import datetime
from pathlib import Path

import click
import pytorch_lightning as L
import torch
from astropy.table import Table

from bliss.datasets.galsim_blends import generate_individual_dataset
from bliss.datasets.lsst import get_default_lsst_psf

NUM_WORKERS = 0

HOME_DIR = Path(__file__).parent.parent.parent
cat = Table.read(HOME_DIR / "data" / "catsim_snr.fits")
CATSIM_TABLE = cat[(cat["i_ab"] < 27.3) & (cat["snr"] > 10)]  # first likely redundant
PSF = get_default_lsst_psf()


@click.command()
@click.option("-s", "--seed", default=42, type=int)
@click.option("-t", "--tag", required=True, type=str, help="Dataset tag")
def main(seed: int, tag: str):

    L.seed_everything(seed)

    train_ds_file = f"/nfs/turbo/lsa-regier/scratch/ismael/datasets/train_ae_ds_{tag}.pt"
    val_ds_file = f"/nfs/turbo/lsa-regier/scratch/ismael/datasets/val_ae_ds_{tag}.pt"
    test_ds_file = f"/nfs/turbo/lsa-regier/scratch/ismael/datasets/test_ae_ds_{tag}.pt"

    if Path(train_ds_file).exists():
        raise IOError("Training file already exists")

    with open("run/log.txt", "a") as f:
        now = datetime.datetime.now()
        print("", file=f)
        log_msg = f"""Run training autoencoder data generation script...
        With tag {tag} and seed {seed} at {now}, n_samples {len(CATSIM_TABLE)}.
        Train, test, and val divided into 3 equal parts on full catalog after
        SNR cut > 10 on catalog.
        """
        print(log_msg, file=f)

    n_rows = len(CATSIM_TABLE)

    # shuffled because of indices in random.choice
    dataset = generate_individual_dataset(n_rows, CATSIM_TABLE, PSF, slen=53, replace=False)

    # train, val, test split
    # no galaxies shared
    train_ds = {p: q[: n_rows // 3] for p, q in dataset.items()}
    val_ds = {p: q[n_rows // 3 : 2 * n_rows // 3] for p, q in dataset.items()}
    test_ds = {p: q[2 * n_rows // 3 :] for p, q in dataset.items()}

    # now save data
    torch.save(train_ds, train_ds_file)
    torch.save(val_ds, val_ds_file)
    torch.save(test_ds, test_ds_file)


if __name__ == "__main__":
    main()
