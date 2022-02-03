import json
import datetime as dt
from typing import Optional
from pathlib import Path

import torch
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


def setup_callbacks(cfg) -> Optional[ModelCheckpoint]:
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
    else:
        checkpoint_callback = None
    return checkpoint_callback


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
    dataset = instantiate(cfg.training.dataset)

    # setup model
    model = instantiate(cfg.training.model, optimizer_params=cfg.training.optimizer_params)

    # setup trainer
    logger = setup_logger(cfg, paths)
    checkpoint_callback = setup_callbacks(cfg)
    profiler = setup_profiler(cfg)

    callbacks = [] if checkpoint_callback is None else [checkpoint_callback]

    trainer = instantiate(
        cfg.training.trainer, logger=logger, profiler=profiler, callbacks=callbacks
    )

    if logger:
        log_hyperparameters(config=cfg, model=model, trainer=trainer)

    # train!
    trainer.fit(model, datamodule=dataset)

    # test!
    if cfg.training.testing.file is not None:
        _ = trainer.test(model, datamodule=dataset)

    # Load best weights from checkpoint
    if cfg.training.weight_save_path is not None:
        model_checkpoint = torch.load(checkpoint_callback.best_model_path, map_location="cpu")
        model_state_dict = model_checkpoint["state_dict"]
        torch.save(model_state_dict, cfg.training.weight_save_path)
        result_path = cfg.training.weight_save_path + ".log.json"
        with open(result_path, "w", encoding="utf-8") as fp:
            cp_data = model_checkpoint["callbacks"][ModelCheckpoint]
            cp_data["timestamp"] = str(dt.datetime.today())
            for k, v in cp_data.items():
                if isinstance(v, torch.Tensor):
                    if not v.shape:
                        cp_data[k] = v.item()
                    else:
                        del cp_data[k]
            fp.write(json.dumps(cp_data))
