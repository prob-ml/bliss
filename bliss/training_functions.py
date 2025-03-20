import datetime
from pathlib import Path

import pytorch_lightning as L
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader, Dataset

from bliss import HOME_DIR
from bliss.datasets.saved_datasets import SavedGalsimBlends, SavedPtiles

NUM_WORKERS = 0

MODELS_DIR = HOME_DIR / "experiment/models"
LOG_FILE = HOME_DIR / "experiment/log.txt"
LOG_FILE_LONG = HOME_DIR / "experiment/log_long.txt"


def setup_training_objects(
    train_ds: Dataset,
    val_ds: Dataset,
    batch_size: int,
    num_workers: int,
    n_epochs: int,
    validate_every_n_epoch: int,
    val_check_interval: float,
    model_name: str,
    log_every_n_steps: int = 16,
    extra_callbacks: list | None = None,  # list of additional callbacks to include
    version: int | str | None = None,  # sometimes specifying version is useful
):
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_dl = DataLoader(val_ds, batch_size=batch_size, num_workers=num_workers)

    mckp = ModelCheckpoint(
        filename="epoch={epoch}-val_loss={val/loss:.3f}",
        save_top_k=5,
        verbose=True,
        monitor="val/loss",
        mode="min",
        save_on_train_epoch_end=False,
        auto_insert_metric_name=False,
    )

    logger = TensorBoardLogger(
        save_dir="out", name=model_name, default_hp_metric=False, version=version
    )

    callbacks = [mckp, *extra_callbacks] if extra_callbacks else [mckp]

    trainer = L.Trainer(
        limit_train_batches=1.0,
        max_epochs=n_epochs,
        logger=logger,
        callbacks=callbacks,
        accelerator="gpu",
        devices=1,
        log_every_n_steps=log_every_n_steps,
        check_val_every_n_epoch=validate_every_n_epoch,
        val_check_interval=val_check_interval,
    )

    return train_dl, val_dl, trainer, logger.version


def run_encoder_training(
    seed: int,
    train_file: str,
    val_file: str,
    batch_size: int,
    n_epochs: int,
    model,
    model_name: str,
    validate_every_n_epoch: int,
    val_check_interval: float,
    log_every_n_steps: int,
    log_info_dict: dict,
    is_deblender=False,
    extra_callbacks: list | None = None,  # list of additional callbacks to include
):
    assert model_name in {"detection", "binary", "deblender"}

    L.seed_everything(seed)

    ds_seed = log_info_dict["ds_seed"]
    assert not (MODELS_DIR / f"{model_name}_{ds_seed}_{seed}.pt").exists(), "model exists."
    if not Path(train_file).exists() or not Path(val_file).exists():
        raise IOError("Training datasets do not exists")

    if is_deblender:
        train_ds = SavedPtiles(train_file)
        val_ds = SavedPtiles(val_file)
    else:
        train_ds = SavedGalsimBlends(train_file)
        val_ds = SavedGalsimBlends(val_file)

    train_dl, val_dl, trainer, vnum = setup_training_objects(
        train_ds=train_ds,
        val_ds=val_ds,
        batch_size=batch_size,
        num_workers=NUM_WORKERS,
        n_epochs=n_epochs,
        validate_every_n_epoch=validate_every_n_epoch,
        val_check_interval=val_check_interval,
        model_name=model_name,
        log_every_n_steps=log_every_n_steps,
        extra_callbacks=extra_callbacks,
    )

    # logging
    log_info_dict.update({"version": vnum})
    _log_info(seed, model_name, log_info_dict)

    # fit!
    trainer.fit(model=model, train_dataloaders=train_dl, val_dataloaders=val_dl)


def _log_info(seed: int, model: str, info: dict):
    now = datetime.datetime.now()

    ds_seed = info["ds_seed"]
    validate_every_n_epoch = info["validate_every_n_epoch"]
    val_check_interval = info["val_check_interval"]
    batch_size = info["batch_size"]
    n_epochs = info["n_epochs"]
    lr = info["lr"]
    train_file = info["train_file"]
    val_file = info["val_file"]
    vnum = info["version"]

    log_msg_short = (
        f"\nTraining {model} with seed {seed}, ds_seed {ds_seed}, version {vnum} at {now}."
    )
    log_msg_long = f"""{log_msg_short}
    validate_every_n_epoch {validate_every_n_epoch},
    val_check_interval {val_check_interval}, batch_size {batch_size}, n_epochs {n_epochs}.
    lr: {lr}

    Using datasets: {train_file}, {val_file}
    """

    ae_path_msg = f"\nAE path: {info['ae_path']}" if model == "deblender" else ""

    log_msg_short += ae_path_msg
    log_msg_long += ae_path_msg

    with open(LOG_FILE, "a", encoding="utf-8") as f:
        print(log_msg_short, file=f)

    with open(LOG_FILE_LONG, "a", encoding="utf-8") as f:
        print("", file=f)
        print(log_msg_long, file=f)
