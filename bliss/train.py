from pathlib import Path

import pytorch_lightning as pl
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.profiler import AdvancedProfiler

from bliss.utils import log_hyperparameters


def setup_seed(cfg):
    if cfg.training.trainer.deterministic:
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
            filename="epoch={epoch}-val_loss={val/loss:.3f}",
            save_top_k=cfg.training.save_top_k,
            verbose=True,
            monitor="val/loss",
            mode="min",
            save_on_train_epoch_end=False,
            auto_insert_metric_name=False,
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
    for key in paths.keys():
        path = Path(paths[key])
        if not path.exists():
            if key == "output":
                path.mkdir(parents=True)
            else:
                raise FileNotFoundError(f"path for {key} ({path.as_posix()}) does not exist")
    setup_seed(cfg)

    # setup dataset.
    dataset = instantiate(cfg.dataset)

    # setup model
    model = instantiate(cfg.model)

    # setup trainer
    logger = setup_logger(cfg, paths)
    callbacks = setup_callbacks(cfg)
    profiler = setup_profiler(cfg)

    trainer = instantiate(
        cfg.training.trainer, logger=logger, profiler=profiler, callbacks=callbacks
    )

    if logger:
        log_hyperparameters(config=cfg, model=model, trainer=trainer)

    # train!
    trainer.fit(model, datamodule=dataset)

    # test!
    if cfg.testing.file is not None:
        _ = trainer.test(model, datamodule=dataset)
