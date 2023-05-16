import datetime as dt
import json
from pathlib import Path
from time import time_ns
from typing import Any, Dict

import pytorch_lightning as pl
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.profiler import AdvancedProfiler
from pytorch_lightning.utilities import rank_zero_only


def train(cfg: DictConfig):  # pylint: disable=too-many-branches, too-many-statements
    # setup paths and seed
    paths = OmegaConf.to_container(cfg.paths, resolve=True)
    assert isinstance(paths, dict)
    for key in paths.keys():
        path = Path(paths[key])
        if not path.exists():
            if key == "output":
                path.mkdir(parents=True)
            else:
                err = "path for {} ({}) does not exist".format(str(key), path.as_posix())
                raise FileNotFoundError(err)
    pl.seed_everything(cfg.training.seed)

    # setup dataset.
    if cfg.training.use_cached_simulator:
        dataset = instantiate(cfg.cached_simulator)
    else:
        dataset = instantiate(cfg.simulator)

    # setup model
    encoder = instantiate(cfg.encoder)

    # load pretrained weights
    if "pretrained_weights" in cfg.training and cfg.training.pretrained_weights is not None:
        enc_state_dict = torch.load(cfg.training.pretrained_weights)
        encoder.load_state_dict(enc_state_dict)

    # setup trainer
    logger = setup_logger(cfg, paths)
    checkpoint_callback = setup_checkpoint_callback(cfg)
    early_stopping = setup_early_stopping(cfg)
    profiler = setup_profiler(cfg)

    callbacks = []
    if checkpoint_callback:
        callbacks.append(checkpoint_callback)
    if early_stopping:
        callbacks.append(early_stopping)

    trainer = instantiate(
        cfg.training.trainer, logger=logger, profiler=profiler, callbacks=callbacks
    )

    if logger:
        log_hyperparameters(config=cfg, trainer=trainer)

    # train!
    tic = time_ns()
    trainer.fit(encoder, datamodule=dataset)
    toc = time_ns()
    train_time_sec = (toc - tic) * 1e-9
    # test!
    if cfg.training.testing.file is not None:
        trainer.test(encoder, datamodule=dataset)

    # Load best weights from checkpoint
    if cfg.training.weight_save_path is not None and (checkpoint_callback is not None):
        model_checkpoint = torch.load(checkpoint_callback.best_model_path, map_location="cpu")
        model_state_dict = model_checkpoint["state_dict"]
        Path(cfg.training.weight_save_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(model_state_dict, cfg.training.weight_save_path)
        result_path = cfg.training.weight_save_path + ".log.json"
        with open(result_path, "w", encoding="utf-8") as fp:
            cp_data = model_checkpoint["callbacks"]
            key = list(cp_data.keys())[0]
            cp_data = cp_data[key]
            data_to_write: Dict[str, Any] = {
                "timestamp": str(dt.datetime.today()),
                "train_time_sec": train_time_sec,
            }
            for k, v in cp_data.items():
                if isinstance(v, torch.Tensor) and not v.shape:
                    data_to_write[k] = v.item()
                elif is_json_serializable(v):
                    data_to_write[k] = v
            fp.write(json.dumps(data_to_write))


def setup_logger(cfg, paths):
    logger = False
    if cfg.training.trainer.logger:
        logger = TensorBoardLogger(
            save_dir=paths["output"],
            name=cfg.training.name,
            version=cfg.training.version,
            default_hp_metric=False,
        )
    return logger


def setup_checkpoint_callback(cfg):
    checkpoint_callback = None
    if cfg.training.trainer.enable_checkpointing:
        checkpoint_callback = ModelCheckpoint(
            filename="epoch={epoch}-val_loss={val/loss:.3f}",
            save_top_k=cfg.training.save_top_k,
            verbose=True,
            monitor="val/loss",
            mode="min",
            save_on_train_epoch_end=False,
            auto_insert_metric_name=False,
        )
    return checkpoint_callback


def setup_early_stopping(cfg):
    early_stopping = None
    if cfg.training.enable_early_stopping:
        early_stopping = EarlyStopping(
            monitor="val/loss", mode="min", patience=cfg.training.patience
        )
    return early_stopping


def setup_profiler(cfg):
    profiler = None
    if cfg.training.trainer.profiler:
        profiler = AdvancedProfiler(filename="profile.txt")
    return profiler


# https://github.com/ashleve/lightning-hydra-template/blob/main/src/utils/utils.py
@rank_zero_only
def log_hyperparameters(config, trainer) -> None:
    """Log config to all Lightning loggers."""
    # send hparams to all loggers
    trainer.logger.log_hyperparams(config)
    # trick to disable logging any more hyperparameters for all loggers
    trainer.logger.log_hyperparams = empty


def empty(*args, **kwargs):
    pass


def is_json_serializable(x):
    ret = True
    try:
        json.dumps(x)
    except TypeError:
        ret = False
    return ret
