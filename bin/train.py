#!/usr/bin/env python3

import hydra
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
import shutil

import torch
import pytorch_lightning as pl
from pytorch_lightning.profiler import AdvancedProfiler
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from bliss import sleep, use_cuda
from bliss.datasets import simulated
from bliss.models import galaxy_net

# compatible datasets and models.
_datamodules = [
    simulated.SimulatedModule,
]
datamodules = {cls.__name__: cls for cls in _datamodules}

_models = [galaxy_net.OneCenteredGalaxy, sleep.SleepPhase]
models = {cls.__name__: cls for cls in _models}


def setup_paths(cfg: DictConfig, enforce_overwrite=True):
    paths = OmegaConf.to_container(cfg.paths, resolve=True)
    output = Path(paths["root"]).joinpath(paths["output"])
    paths["output"] = output.as_posix()

    if enforce_overwrite:
        assert not output.exists() or cfg.general.overwrite, "Enforcing overwrite."
        if cfg.general.overwrite and output.exists():
            shutil.rmtree(output)

    output.mkdir(parents=False, exist_ok=not enforce_overwrite)

    for p in paths.values():
        assert Path(p).exists(), f"path {p.as_posix()} does not exist"

    return paths


def setup_seed(cfg):
    if cfg.training.deterministic:
        assert cfg.training.seed is not None
        pl.seed_everything(cfg.training.seed)


def setup_profiler(cfg, paths):
    profiler = False
    output = Path(paths["output"])
    if cfg.training.trainer.profiler:
        profile_file = output.joinpath("profile.txt")
        profiler = AdvancedProfiler(output_filename=profile_file)
    return profiler


def setup_logger(cfg, paths):
    logger = False
    if cfg.training.trainer.logger:
        logger = TensorBoardLogger(save_dir=paths["output"], name="lightning_logs")
    return logger


def setup_checkpoint_callback(cfg, paths, logger):
    checkpoint_callback = False
    output = Path(paths["output"])
    if cfg.training.trainer.checkpoint_callback:
        checkpoint_dir = f"lightning_logs/version_{logger.version}/checkpoints"
        checkpoint_dir = output.joinpath(checkpoint_dir)
        checkpoint_callback = ModelCheckpoint(
            filepath=checkpoint_dir,
            save_top_k=True,
            verbose=True,
            monitor="val_loss",
            mode="min",
            prefix="",
        )

    return checkpoint_callback


@hydra.main(config_path="../config", config_name="config")
def main(cfg: DictConfig):

    # setup gpus
    gpus = cfg.training.trainer.gpus
    if gpus and use_cuda:
        assert gpus[1] == "," and len(gpus) == 2, "Format accepted: 'Y,' "
        device_id = gpus[0]
        device = torch.device(f"cuda:{device_id}")
        torch.cuda.set_device(device)

    # setup paths and seed
    paths = setup_paths(cfg, enforce_overwrite=False)
    setup_seed(cfg)

    # setup dataset.
    datamodule = datamodules[cfg.dataset.name](cfg)

    # setup model
    model = models[cfg.model.name](cfg)

    # setup trainer
    profiler = setup_profiler(cfg, paths)
    logger = setup_logger(cfg, paths)
    checkpoint_callback = setup_checkpoint_callback(cfg, paths, logger)
    trainer_dict = OmegaConf.to_container(cfg.training.trainer, resolve=True)
    trainer_dict.update(
        dict(logger=logger, profiler=profiler, checkpoint_callback=checkpoint_callback)
    )
    trainer = pl.Trainer(**trainer_dict)

    # train!
    trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":
    main()
