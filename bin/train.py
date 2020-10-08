#!/usr/bin/env python3

import hydra
from omegaconf import DictConfig
import torch
import pytorch_lightning as pl
from pytorch_lightning.profiler import AdvancedProfiler
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from .utils import setup_paths

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


@hydra.main(config_path="config", config_name="config")
def main(cfg: DictConfig):

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

    # train!
    trainer.fit(model)


if __name__ == "__main__":
    main()
