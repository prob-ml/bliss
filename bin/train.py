#!/usr/bin/env python3

import argparse
import os
import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning.profiler import AdvancedProfiler
from pytorch_lightning.loggers import TensorBoardLogger

from . import setup_paths, setup_device

from celeste import use_cuda
from celeste.datasets import galaxy_datasets
from celeste.models import galaxy_net

datasets = [galaxy_datasets.H5Catalog, galaxy_datasets.CatsimGalaxies]
datasets = {cls.__name__: cls for cls in datasets}

models = [galaxy_net.OneCenteredGalaxy]
models = {cls.__name__: cls for cls in models}


def setup_seed(args):
    if args.torch_seed:
        torch.manual_seed(args.torch_seed)

        if use_cuda:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    if args.numpy_seed:
        np.random.seed(args.numpy_seed)


def setup_profiler(args, output_path):
    profiler = None
    if args.profile:
        profile_file = output_path.joinpath("profile.txt")
        profiler = AdvancedProfiler(output_filename=profile_file)
    return profiler


def setup_logger(args, output_path):
    logger = None
    if args.save_logs:
        logger = TensorBoardLogger(
            save_dir=output_path, version=1, name="lightning_logs"
        )
    return logger


def main(args):

    assert args.model in models, "Not implemented."

    # setup.
    paths = setup_paths(args)
    setup_device(args)
    setup_seed(args)
    output_path = paths["results"].joinpath(args.output_dir)

    # setup dataset.
    dataset = datasets[args.dataset].from_args(args)

    # setup model
    model_cls = models[args.model]
    model = model_cls.from_args(dataset, args)

    # setup trainer
    profiler = setup_profiler(args, output_path)
    logger = setup_logger(args, output_path)
    n_device = [args.device]
    sleep_trainer = pl.Trainer(
        gpus=n_device,
        profiler=profiler,
        min_epochs=args.n_epochs,
        max_epochs=args.n_epochs,
        reload_dataloaders_every_epoch=True,
        default_root_dir=output_path,
        logger=logger,
    )

    # train!
    sleep_trainer.fit(model)


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
    # Profile and Logging
    # ----------------

    parser.add_argument("--profile", action="store_true", help="Whether to profile.")
    parser.add_argument("--save-log", action="store_true", help="Log output?")

    # ---------------
    # Paths
    # ----------------
    parser.add_argument(
        "--root-dir",
        help="Absolute path to directory containing bin and celeste package.",
        type=str,
        default=os.path.abspath("."),
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
        choices=[*models],
        required=True,
        help="What are we training?",
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
        choices=[*datasets],
        required=True,
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
