import datetime as dt
import json
from pathlib import Path
from time import time_ns
from typing import Any, Dict

import pytorch_lightning as pl
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.profilers import AdvancedProfiler
from pytorch_lightning.utilities import rank_zero_only


def train(train_cfg: DictConfig):  # pylint: disable=too-many-branches, too-many-statements
    # setup seed
    pl.seed_everything(train_cfg.seed)

    # setup dataset
    data_source_cfg = train_cfg.data_source
    dataset = instantiate(data_source_cfg)

    # setup model
    encoder = instantiate(train_cfg.encoder)

    # load pretrained weights
    if train_cfg.pretrained_weights is not None:
        enc_state_dict = torch.load(train_cfg.pretrained_weights)
        encoder.load_state_dict(enc_state_dict)

    # setup logger
    logger = False
    if train_cfg.trainer.logger:
        logger = TensorBoardLogger(
            save_dir=train_cfg.output_dir,
            name=train_cfg.name,
            version=train_cfg.version,
            default_hp_metric=False,
        )

    # setup trainer
    checkpoint_callback = setup_checkpoint_callback(train_cfg)
    early_stopping = setup_early_stopping(train_cfg)
    profiler = setup_profiler(train_cfg)

    callbacks = []
    if checkpoint_callback:
        callbacks.append(checkpoint_callback)
    if early_stopping:
        callbacks.append(early_stopping)

    trainer = instantiate(train_cfg.trainer, logger=logger, profiler=profiler, callbacks=callbacks)

    if logger:
        log_hyperparameters(train_cfg=train_cfg, trainer=trainer)

    # train!
    tic = time_ns()
    trainer.fit(encoder, datamodule=dataset)
    toc = time_ns()
    train_time_sec = (toc - tic) * 1e-9
    # test!
    if train_cfg.testing:
        trainer.test(encoder, datamodule=dataset)

    # Save the best model to the final path
    if train_cfg.weight_save_path and checkpoint_callback.best_model_path:
        save_best_model(train_cfg, checkpoint_callback.best_model_path, train_time_sec)


def setup_checkpoint_callback(train_cfg):
    checkpoint_callback = None
    if train_cfg.trainer.enable_checkpointing:
        checkpoint_callback = ModelCheckpoint(
            filename="epoch{epoch}",
            save_top_k=train_cfg.save_top_k,
            verbose=True,
            monitor="val-layer1/_loss",
            mode="min",
            save_on_train_epoch_end=False,
            auto_insert_metric_name=False,
        )
    return checkpoint_callback


def setup_early_stopping(train_cfg):
    early_stopping = None
    if train_cfg.enable_early_stopping:
        early_stopping = EarlyStopping(
            monitor="val-layer1/_loss", mode="min", patience=train_cfg.patience
        )
    return early_stopping


def setup_profiler(train_cfg):
    profiler = None
    if train_cfg.trainer.profiler:
        profiler = AdvancedProfiler(filename="profile")
    return profiler


# https://github.com/ashleve/lightning-hydra-template/blob/main/src/utils/utils.py
@rank_zero_only
def log_hyperparameters(train_cfg, trainer) -> None:
    """Log config to all Lightning loggers."""
    # send hparams to all loggers
    trainer.logger.log_hyperparams(train_cfg)
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


def save_best_model(train_cfg, best_model_path, train_time_sec):
    """Saves the best checkpoint to the final model path."""
    # load best weights from checkpoint
    model_checkpoint = torch.load(best_model_path, map_location="cpu")
    model_state_dict = model_checkpoint["state_dict"]

    # save model
    Path(train_cfg.weight_save_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model_state_dict, train_cfg.weight_save_path)

    # save log
    result_path = train_cfg.weight_save_path + ".log.json"
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
