#!/usr/bin/env python3

import hydra
from omegaconf import DictConfig, OmegaConf

import torch
import pytorch_lightning as pl
from pytorch_lightning.profiler import AdvancedProfiler
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from .utils import setup_paths

from bliss import sleep, use_cuda
from bliss.datasets import galaxy_datasets, catsim, simulated
from bliss.models import galaxy_net

# compatible datasets and models.
_datasets = [
    galaxy_datasets.H5Catalog,
    catsim.CatsimGalaxies,
    simulated.SimulatedDataset,
]
datasets = {cls.__name__: cls for cls in _datasets}

_models = [galaxy_net.OneCenteredGalaxy, sleep.SleepPhase]
models = {cls.__name__: cls for cls in _models}


def setup_seed(cfg):
    if cfg.training.deterministic:
        assert cfg.training.seed is not None
        pl.seed_everything(cfg.training.seed)


def setup_profiler(cfg, paths):
    profiler = False
    if cfg.trainer.profiler:
        profile_file = paths["output"].joinpath("profile.txt")
        profiler = AdvancedProfiler(output_filename=profile_file)
    return profiler


def setup_logger(cfg, paths):
    logger = False
    if cfg.trainer.logger:
        logger = TensorBoardLogger(save_dir=paths["output"], name="lightning_logs")
    return logger


def setup_checkpoint_callback(cfg, paths, logger):
    checkpoint_callback = False
    if cfg.trainer.checkpoint_callback:
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


@hydra.main(config_path="../config", config_name="config")
def main(cfg: DictConfig):

    # setup gpus
    gpus = cfg.general.gpus
    if gpus and use_cuda:
        assert gpus[1] == "," and len(gpus) == 2, "Format accepted: 'Y,' "
        device_id = gpus[0]
        device = torch.device(f"cuda:{device_id}")
        torch.cuda.set_device(device)

    # setup.
    paths = setup_paths(cfg, enforce_overwrite=False)
    setup_seed(cfg)

    # setup dataset.
    dataset = datasets[cfg.dataset.class_name](cfg)

    # setup model
    model = models[cfg.model.class_name](hparams={}, cfg=cfg, dataset=dataset)

    # setup trainer
    profiler = setup_profiler(cfg, paths)
    logger = setup_logger(cfg, paths)
    checkpoint_callback = setup_checkpoint_callback(cfg, paths, logger)
    trainer_dict = OmegaConf.to_container(cfg.trainer, resolve=True)
    trainer_dict.update(
        dict(logger=logger, profiler=profiler, checkpoint_callback=checkpoint_callback)
    )
    trainer = pl.Trainer(**trainer_dict)

    # train!
    trainer.fit(model)


if __name__ == "__main__":
    main()
