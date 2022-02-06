import datetime as dt
import json
from pathlib import Path
from typing import Optional

import pytorch_lightning as pl
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.profiler import AdvancedProfiler
from pytorch_lightning.utilities import rank_zero_only


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
            cp_data = model_checkpoint["callbacks"]
            key = list(cp_data.keys())[0]
            cp_data = cp_data[key]
            data_to_write = {"timestamp": str(dt.datetime.today())}
            for k, v in cp_data.items():
                if isinstance(v, torch.Tensor) and not v.shape:
                    data_to_write[k] = v.item()
                elif is_json_serializable(v):
                    data_to_write[k] = v
            fp.write(json.dumps(data_to_write))


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
    else:
        checkpoint_callback = None
    return checkpoint_callback


def setup_profiler(cfg):
    profiler = None
    if cfg.training.trainer.profiler:
        profiler = AdvancedProfiler(filename="profile.txt")
    return profiler


# https://github.com/ashleve/lightning-hydra-template/blob/main/src/utils/utils.py
@rank_zero_only
def log_hyperparameters(config, model, trainer) -> None:
    """Log config and num of model parameters to all Lightning loggers."""

    hparams = {}

    # choose which parts of hydra config will be saved to loggers
    hparams["mode"] = config["mode"]
    hparams["gpus"] = config["gpus"]
    hparams["training"] = config["training"]
    hparams["model"] = config["training"]["model"]
    hparams["dataset"] = config["training"]["dataset"]
    hparams["optimizer"] = config["training"]["optimizer_params"]

    # save number of model parameters
    hparams["model/params_total"] = sum(p.numel() for p in model.parameters())
    hparams["model/params_trainable"] = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    hparams["model/params_not_trainable"] = sum(
        p.numel() for p in model.parameters() if not p.requires_grad
    )

    # send hparams to all loggers
    trainer.logger.log_hyperparams(hparams)

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
