#!/usr/bin/env python3

import subprocess

import pytorch_lightning as L
import typer

from experiment import DATASETS_DIR, SEED


def main(
    single: bool = False,
    blends: bool = False,
    tiles: bool = False,
    centrals: bool = False,
    all: bool = False,
):
    L.seed_everything(SEED)
    DATASETS_DIR.mkdir(exist_ok=True)  # create dataset path if does not exist
    indices_fname = f"indices_{SEED}.npz"

    if single or all:
        cmd = f"./scripts/datasets/get_single_galaxies_dataset.py --seed {SEED}"
        subprocess.check_call(cmd, shell=True)

    if blends or all:
        cmd = f"./scripts/datasets/get_blends_dataset.py --seed {SEED} --indices-fname {indices_fname}"
        subprocess.check_call(cmd, shell=True)

    if tiles or all:
        cmd = (
            f"./scripts/datasets/get_tiles_dataset.py --seed {SEED} --indices-fname {indices_fname}"
        )
        subprocess.check_call(cmd, shell=True)

    if centrals or all:
        cmd = f"./scripts/datasets/get_centrals_dataset.py --seed {SEED} --indices-fname {indices_fname}"
        subprocess.check_call(cmd, shell=True)


if __name__ == "__main__":
    typer.run(main)
