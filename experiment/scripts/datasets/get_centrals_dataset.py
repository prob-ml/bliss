#!/usr/bin/env python3

import numpy as np
import pytorch_lightning as L
import typer

from bliss.datasets.central_sim import generate_central_sim_dataset
from bliss.datasets.io import save_dataset_npz
from bliss.datasets.lsst import (
    get_default_lsst_psf,
    prepare_final_galaxy_catalog,
)
from experiment import DATASETS_DIR


def main(
    seed: int = typer.Option(),
    indices_fname: str = typer.Option(),
    n_images: int = 10_000,
    slen: int = 35,
):
    L.seed_everything(seed)

    dataset_path = DATASETS_DIR / f"central_ds_{seed}.npz"
    indices_dict = np.load(DATASETS_DIR / indices_fname)
    test_indices = indices_dict["test"]

    assert not dataset_path.exists(), "Already exists."

    cat = prepare_final_galaxy_catalog()
    psf = get_default_lsst_psf()
    print(f"Number of test galaxies: {len(cat[test_indices])}")
    ds = generate_central_sim_dataset(
        n_samples=n_images,
        catsim_table=cat[test_indices],
        psf=psf,
        slen=slen,
        max_n_sources=10,
        mag_cut_central=25.3,
        bp=24,
    )
    save_dataset_npz(ds, dataset_path)


if __name__ == "__main__":
    typer.run(main)
