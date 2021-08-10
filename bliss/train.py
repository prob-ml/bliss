from pathlib import Path

import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.profiler import AdvancedProfiler

from bliss import sleep
from bliss.datasets import galsim_galaxies, simulated
from bliss.models import galaxy_encoder, galaxy_net

# available datasets and models.
_datasets = [
    simulated.SimulatedDataset,
    galsim_galaxies.ToyGaussian,
    galsim_galaxies.SDSSGalaxies,
    galsim_galaxies.SavedGalaxies,
]
datasets = {cls.__name__: cls for cls in _datasets}

_models = [sleep.SleepPhase, galaxy_net.OneCenteredGalaxyAE, galaxy_encoder.GalaxyEncoder]
models = {cls.__name__: cls for cls in _models}


def setup_seed(cfg):
    if cfg.training.deterministic:
        assert cfg.training.seed is not None
        pl.seed_everything(cfg.training.seed)


def setup_logger(cfg, paths):
    logger = False
    if cfg.training.trainer.logger:
        logger = TensorBoardLogger(
            save_dir=paths["output"],
            name=cfg.training.experiment,
            version=cfg.training.version,
            default_hp_metric=False,
        )
    return logger


def setup_callbacks(cfg):
    callbacks = []
    if cfg.training.trainer.checkpoint_callback:
        checkpoint_callback = ModelCheckpoint(
            filename="{epoch}-{val/loss:.2f}",
            save_top_k=cfg.training.save_top_k,
            verbose=True,
            monitor="val/loss",
            mode="min",
            save_on_train_epoch_end=False,
        )
        callbacks.append(checkpoint_callback)

    return callbacks


def setup_profiler(cfg):
    profiler = None
    if cfg.training.trainer.profiler:
        profiler = AdvancedProfiler(filename="profile.txt")
    return profiler


def train(cfg: DictConfig):

    # setup paths and seed
    paths = OmegaConf.to_container(cfg.paths, resolve=True)
    for p in paths.values():
        assert Path(p).exists(), f"path {Path(p).as_posix()} does not exist"
    setup_seed(cfg)

    # setup dataset.
    dataset = datasets[cfg.dataset.name](**cfg.dataset.kwargs)

    # setup model
    model = models[cfg.model.name](**cfg.model.kwargs, optimizer_params=cfg.optimizer)

    # setup trainer
    logger = setup_logger(cfg, paths)
    callbacks = setup_callbacks(cfg)
    profiler = setup_profiler(cfg)
    trainer_dict = OmegaConf.to_container(cfg.training.trainer, resolve=True)
    trainer_dict.update(dict(logger=logger, profiler=profiler, callbacks=callbacks))
    trainer = pl.Trainer(**trainer_dict)

    # train!
    trainer.fit(model, datamodule=dataset)

    # test!
    if cfg.testing.file is not None:
        _ = trainer.test(model, datamodule=dataset)
