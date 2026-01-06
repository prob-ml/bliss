#!/usr/bin/env python3
import subprocess

import pytorch_lightning as L
import typer

from experiment import CACHE_DIR, FIGURE_DIR, SEED


def main(
    detection: bool = False,
    binary: bool = False,
    deblend: bool = False,
    toy: bool = False,
    samples: bool = False,
    all: bool = False,
    overwrite: bool = False,
):
    L.seed_everything(SEED)
    FIGURE_DIR.mkdir(exist_ok=True)
    CACHE_DIR.mkdir(exist_ok=True)
    overwrite_txt = "--overwrite" if overwrite else "--no-overwrite"

    if detection or all:
        cmd = f"./scripts/get_figures.py --mode detection {overwrite_txt}"
        subprocess.check_call(cmd, shell=True)

    if binary or all:
        cmd = f"./scripts/get_figures.py --mode binary {overwrite_txt}"
        subprocess.check_call(cmd, shell=True)

    if deblend or all:
        cmd = f"./scripts/get_figures.py --mode deblend {overwrite_txt}"
        subprocess.check_call(cmd, shell=True)

    if toy or all:
        cmd = f"./scripts/get_figures.py --mode toy {overwrite_txt}"
        subprocess.check_call(cmd, shell=True)

    if samples or all:
        cmd = f"./scripts/get_figures.py --mode samples {overwrite_txt}"
        subprocess.check_call(cmd, shell=True)


if __name__ == "__main__":
    typer.run(main)
