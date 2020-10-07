#!/usr/bin/env python3

import argparse
import torch
import pytorch_lightning as pl
from pytorch_lightning.profiler import AdvancedProfiler
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from .utils import setup_paths, add_path_args

from bliss import sleep
from bliss.datasets import galaxy_datasets, catsim, simulated
from bliss.models import galaxy_net

_datasets = [
    galaxy_datasets.H5Catalog,
    catsim.CatsimGalaxies,
    simulated.SimulatedDataset,
]
datasets = {cls.__name__: cls for cls in _datasets}

_models = [galaxy_net.OneCenteredGalaxy, sleep.SleepPhase]
models = {cls.__name__: cls for cls in _models}


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
        logger = TensorBoardLogger(save_dir=paths["output"], name="lightning_logs")
    return logger


def setup_checkpoint_callback(args, paths, logger):
    checkpoint_callback = False
    if args.checkpoint_callback:
        checkpoint_dir = f"lightning_logs/version_{logger.version}/checkpoints"
        checkpoint_dir = paths["output"].joinpath(checkpoint_dir)
        checkpoint_callback = ModelCheckpoint(
            filepath=checkpoint_dir,
            save_top_k=True,
            verbose=True,
            monitor="val_loss",
            mode="min",
            prefix="",
        )

    return checkpoint_callback


def main(args):

    # setup gpus
    if args.gpus:
        assert args.gpus[1] == "," and len(args.gpus) == 2, "Format accepted: 'Y,' "
        device_str = args.gpus[0]
        device_id = int(device_str)
        device = torch.device(f"cuda:{device_id}")
        torch.cuda.set_device(device)

    # setup.
    paths = setup_paths(args, enforce_overwrite=False)
    setup_seed(args)

    # setup dataset.
    dataset = datasets[args.dataset].from_args(args, paths)

    # setup model
    model_cls = models[args.model]
    model = model_cls.from_args(args, dataset)

    # setup trainer
    profiler = setup_profiler(args, paths)
    logger = setup_logger(args, paths)
    checkpoint_callback = setup_checkpoint_callback(args, paths, logger)
    trainer = pl.Trainer.from_argparse_args(
        args, logger=logger, profiler=profiler, checkpoint_callback=checkpoint_callback
    )

    if args.dry_run:
        print("dataset params:", vars(dataset))
        print()
        if hasattr(dataset, "image_decoder"):
            print("decoder params:", dataset.image_decoder.get_props())
            print()

        if hasattr(model, "image_encoder"):
            print("encoder params:", model.image_encoder.get_props())
        return

    # train!
    trainer.fit(model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Training model [argument parser]",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for pytorch, numpy, ...",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Check if parameters are correct " "and do nothing else.",
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
        "--model",
        type=str,
        choices=[*models],
        required=True,
        help="What are we training?",
    )
    models_group.add_argument("--slen", type=int, default=51)
    models_group.add_argument("--n-bands", type=int, default=1)
    models_group.add_argument("--latent-dim", type=int, default=8, help="For galaxies")
    models_group.add_argument(
        "--tile-slen",
        type=int,
        default=2,
        help="Distance between tile centers in pixels.",
    )

    # one centered galaxy
    one_centered_galaxy_group = parser.add_argument_group("[One Centered Galaxy Model]")
    galaxy_net.OneCenteredGalaxy.add_args(one_centered_galaxy_group)

    # sleep image encoder
    image_encoder_group = parser.add_argument_group("[Sleep Phase Image Encoder]")
    sleep.SleepPhase.add_args(image_encoder_group)

    # ---------------
    # Dataset
    # ----------------
    general_dataset_group = parser.add_argument_group("[All Datasets]")
    general_dataset_group.add_argument(
        "--dataset",
        type=str,
        choices=[*datasets],
        required=True,
        help="Specifies the dataset to be used to train the model.",
    )

    general_dataset_group.add_argument(
        "--n-batches",
        type=int,
        default=1,
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
    catsim.CatsimGalaxies.add_args(catsim_group)

    # simulated
    simulated_group = parser.add_argument_group("[Simulated Dataset]")
    simulated.SimulatedDataset.add_args(simulated_group)

    # ---------------
    # Trainer
    # ----------------
    parser = pl.Trainer.add_argparse_args(parser)
    pargs = parser.parse_args()

    main(pargs)
