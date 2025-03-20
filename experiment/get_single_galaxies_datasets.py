#!/usr/bin/env python3

import datetime

import numpy as np
import pytorch_lightning as L
import typer

from bliss import DATASETS_DIR, HOME_DIR
from bliss.datasets.generate_individual import generate_individual_dataset
from bliss.datasets.io import save_dataset_npz
from bliss.datasets.lsst import get_default_lsst_psf, prepare_final_galaxy_catalog

NUM_WORKERS = 0

LOG_FILE = HOME_DIR / "experiment/log.txt"

CATSIM_CAT = prepare_final_galaxy_catalog()
PSF = get_default_lsst_psf()


def main(seed: int = typer.Option(), fraction: float = 1.0):
    L.seed_everything(seed)
    rng = np.random.default_rng(seed)  # for catalog indices

    train_ds_file = DATASETS_DIR / f"train_ae_ds_{seed}.npz"
    val_ds_file = DATASETS_DIR / f"val_ae_ds_{seed}.npz"
    test_ds_file = DATASETS_DIR / f"test_ae_ds_{seed}.npz"

    assert not train_ds_file.exists(), "files exist"
    assert not val_ds_file.exists(), "files exist"
    assert not test_ds_file.exists(), "files exist"

    n_rows = len(CATSIM_CAT)
    shuffled_indices = rng.choice(np.arange(n_rows), size=n_rows, replace=False)
    train_indices = shuffled_indices[: n_rows // 3]
    val_indices = shuffled_indices[n_rows // 3 : n_rows // 3 * 2]
    test_indices = shuffled_indices[n_rows // 3 * 2 :]

    # save indices, will reuse for blends.
    np.savez(
        DATASETS_DIR / f"indices_{seed}.npz",
        train=train_indices,
        val=val_indices,
        test=test_indices,
    )

    all_files = (train_ds_file, val_ds_file, test_ds_file)
    all_indices = (train_indices, val_indices, test_indices)
    for fpath, idxs in zip(all_files, all_indices, strict=True):
        cat = CATSIM_CAT[idxs]
        n_samples = int(len(cat) * fraction)
        ds = generate_individual_dataset(n_samples, cat, PSF, replace=False)
        save_dataset_npz(ds, fpath)

    # logging
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        now = datetime.datetime.now()
        log_msg = (
            f"\nRun single galaxy data generation script with seed {seed}, "
            f"fraction{fraction} at {now}."
        )
        print(log_msg, file=f)


if __name__ == "__main__":
    typer.run(main)
