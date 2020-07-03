#!/usr/bin/env python3

import argparse
import os
import pytorch_lightning as pl
from pytorch_lightning.profiler import AdvancedProfiler
from pytorch_lightning.loggers import TensorBoardLogger

from . import setup_paths, setup_device, add_path_args

from celeste.datasets import galaxy_datasets
from celeste.models import galaxy_net

datasets = [galaxy_datasets.H5Catalog, galaxy_datasets.CatsimGalaxies]
datasets = {cls.__name__: cls for cls in datasets}

models = [galaxy_net.OneCenteredGalaxy]
models = {cls.__name__: cls for cls in models}


def setup_seed(args):
    if args.deterministic:
        assert args.seed is not None
        pl.seed_everything(args.seed)


def setup_profiler(args, paths):
    profiler = None
    if args.profiler:
        profile_file = paths["output"].joinpath("profile.txt")
        profiler = AdvancedProfiler(output_filename=profile_file)
    return profiler


def setup_logger(args, paths):
    logger = False
    if args.logger:
        logger = TensorBoardLogger(
            save_dir=paths["output"], version=1, name="lightning_logs"
        )
    return logger


def main(args):

    assert args.model_name in models, "Not implemented."

    # setup.
    paths = setup_paths(args)
    setup_device(args)
    setup_seed(args)

    # setup dataset.
    dataset = datasets[args.dataset_name].from_args(args)

    # setup model
    model_cls = models[args.model_name]
    model = model_cls.from_args(dataset, args)

    # setup trainer
    profiler = setup_profiler(args, paths)
    logger = setup_logger(args, paths)
    trainer = pl.Trainer.from_argparse_args(args, logger=logger, profiler=profiler)

    # train!
    trainer.fit(model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Training model [argument parser]",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--seed", type=int, default=None, help="Random seed for pytorch, numpy, ...",
    )

    # ---------------
    # Paths
    # ----------------
    parser = add_path_args(parser)

    # ---------------
    # Optimizer
    # ----------------
    optimizer_group = parser.add_argument_group("[Optimizer]")
    optimizer_group.add_argument("--lr", type=float, default=1e-4)
    optimizer_group.add_argument("--weight-decay", type=float, default=1e-6)

    # ---------------
    # Model
    # ----------------
    models_group = parser.add_argument_group("[All Models]")
    models_group.add_argument(
        "--model-name",
        type=str,
        choices=[*models],
        required=True,
        help="What are we training?",
    )
    models_group.add_argument("--slen", type=int, default=51)
    models_group.add_argument("--n-bands", type=int, default=1)

    # one centered galaxy
    one_centered_galaxy_group = parser.add_argument_group("[One Centered Galaxy Model]")
    galaxy_net.OneCenteredGalaxy.add_args(one_centered_galaxy_group)

    # ---------------
    # Dataset
    # ----------------
    general_dataset_group = parser.add_argument_group("[All Datasets]")
    general_dataset_group.add_argument(
        "--dataset-name",
        type=str,
        choices=[*datasets],
        required=True,
        help="Specifies the dataset to be used to train the model.",
    )

    general_dataset_group.add_argument(
        "--batch-size",
        type=int,
        default=64,
        metavar="BS",
        help="input batch size for training.",
    )
    general_dataset_group.add_argument("--num-workers", type=int, default=0)

    # h5
    h5_group = parser.add_argument_group("[H5 Dataset]")
    galaxy_datasets.H5Catalog.add_args(h5_group)

    # catsim galaxies
    catsim_group = parser.add_argument_group("[Catsim Dataset]")
    galaxy_datasets.CatsimGalaxies.add_args(catsim_group)

    # ---------------
    # Trainer
    # ----------------
    parser = pl.Trainer.add_argparse_args(parser)
    pargs = parser.parse_args()

    main(pargs)
