from pathlib import Path
import shutil

import os
from omegaconf import DictConfig, OmegaConf

import pytorch_lightning as pl
from pytorch_lightning.profiler import AdvancedProfiler
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from bliss import sleep
from bliss.datasets import simulated, galsim_galaxies
from bliss.models import galaxy_net

# available datasets and models.
_datasets = [
    simulated.SimulatedDataset,
    galsim_galaxies.ToyGaussian,
    galsim_galaxies.SDSSGalaxies,
    galsim_galaxies.SavedGalaxies,
]
datasets = {cls.__name__: cls for cls in _datasets}

_models = [sleep.SleepPhase, galaxy_net.OneCenteredGalaxyAE]
models = {cls.__name__: cls for cls in _models}


def setup_paths(cfg: DictConfig, enforce_overwrite=True):
    paths = OmegaConf.to_container(cfg.paths, resolve=True)
    output = Path(paths["root"]).joinpath(paths["output"])
    paths["output"] = output.as_posix()
    if not os.path.exists(paths["output"]):
        os.makedirs(paths["output"])

    if enforce_overwrite:
        assert not output.exists() or cfg.general.overwrite, "Enforcing overwrite."
        if cfg.general.overwrite and output.exists():
            shutil.rmtree(output)

    output.mkdir(parents=False, exist_ok=not enforce_overwrite)

    for p in paths.values():
        assert Path(p).exists(), f"path {Path(p).as_posix()} does not exist"

    return paths


def setup_seed(cfg):
    if cfg.training.deterministic:
        assert cfg.training.seed is not None
        pl.seed_everything(cfg.training.seed)


def setup_profiler(cfg, paths):
    profiler = None
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
        checkpoint_dir = f"lightning_logs/version_{logger.version}"
        checkpoint_dir = output.joinpath(checkpoint_dir)
        checkpoint_callback = ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename="{epoch}",
            save_top_k=1,
            verbose=True,
            monitor="val_loss",
            mode="min",
        )

    return checkpoint_callback


def train(cfg: DictConfig):

    # setup paths and seed
    paths = setup_paths(cfg, enforce_overwrite=False)
    setup_seed(cfg)

    # setup dataset.
    dataset = datasets[cfg.dataset.name](**cfg.dataset.kwargs)

    # setup model
    model = models[cfg.model.name](**cfg.model.kwargs, optimizer_params=cfg.optimizer)

    # setup trainer
    profiler = setup_profiler(cfg, paths)
    logger = setup_logger(cfg, paths)
    checkpoint_callback = setup_checkpoint_callback(cfg, paths, logger)
    trainer_dict = OmegaConf.to_container(cfg.training.trainer, resolve=True)
    trainer_dict.update(dict(logger=logger, profiler=profiler, callbacks=[checkpoint_callback]))
    trainer = pl.Trainer(**trainer_dict)

    # train!
    trainer.fit(model, datamodule=dataset)

    # test!
    if cfg.testing.file is not None:
        _ = trainer.test(model, datamodule=dataset)
