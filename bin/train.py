#!/usr/bin/env python3

import argparse
import numpy as np
import subprocess
from pathlib import Path
import torch
import os

from . import setup_paths, setup_device
from celeste import use_cuda
from celeste.datasets import galaxy_datasets
from celeste.models import galaxy_net

datasets = [galaxy_datasets.H5Catalog, galaxy_datasets.CatsimGalaxies]
datasets = {cls.__name__: cls for cls in datasets}

models = [galaxy_net.OneCenteredGalaxy]
models = {cls.__name__: cls for cls in models}


def main(args):

    assert args.model in models, "Not implemented."

    # setup paths.
    setup_paths(args)
    setup_device(args)

    # setup dataset.

    # setup additional arguments for model.

    # create model.
    model = models[args.model].from_args()

    # setup trainer

    # train!
    # Additional settings.
    args["dir_name"] = testing_path.joinpath(args["dir_name"])
    project_dir = Path(args["dir_name"])

    # check if directory exists or if we should overwrite.
    if project_dir.is_dir() and not args["overwrite"]:
        raise IOError("Directory already exists.")
    elif project_dir.is_dir():
        subprocess.run(f"rm -r {project_dir.as_posix()}", shell=True)

    project_dir.mkdir()

    torch.manual_seed(pargs.seed)
    np.random.seed(pargs.seed)

    # run.
    with torch.cuda.device(args["device"]):
        run(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Training model [argument parser]",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ---------------
    # Device
    # ----------------
    parser.add_argument(
        "--device", type=int, default=0, metavar="DEV", help="GPU device ID"
    )

    parser.add_argument(
        "--no-cuda",
        action="store_true",
        help="whether to using a discrete graphics card",
    )

    parser.add_argument(
        "--torch-seed", type=int, default=None, help="Random seed for pytorch",
    )
    parser.add_argument(
        "--np-seed", type=int, default=None, help="Random seed for numpy",
    )

    # ---------------
    # Paths
    # ----------------
    parser.add_argument(
        "--root-dir",
        help="Absolute path to directory containing bin and celeste package.",
        type=str,
        default=os.path.abspath(".."),
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory name relative to root/results path, where output will be saved.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Whether to overwrite if directory already exists.",
    )

    # ---------------
    # Training
    # ----------------
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        metavar="BS",
        help="input batch size for training.",
    )
    parser.add_argument(
        "--n-epochs",
        type=int,
        default=100,
        metavar="E",
        help="number of epochs to train.",
    )
    parser.add_argument(
        "--evaluate-every",
        type=int,
        default=None,
        help="Whether to evaluate and log every so epochs.",
    )

    # ---------------
    # Model
    # ----------------
    parser.add_argument(
        "--model",
        type=str,
        help="What model we are training?",
        choices=[*models],
        required=True,
    )

    # one centered galaxy
    one_centered_galaxy_group = parser.add_argument_group("[One Centered Galaxy Model]")
    galaxy_net.OneCenteredGalaxy.add_args(one_centered_galaxy_group)

    # ---------------
    # Dataset
    # ----------------
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=[*datasets],
        help="Specifies the dataset to be used to train the model.",
    )

    # h5
    h5_group = parser.add_argument_group("[H5 Dataset]")
    galaxy_datasets.H5Catalog.add_args(h5_group)

    # catsim galaxies
    catsim_group = parser.add_argument_group("[Catsim Dataset]")
    galaxy_datasets.CatsimGalaxies.add_args(catsim_group)

    pargs = parser.parse_args()

    main(pargs)
