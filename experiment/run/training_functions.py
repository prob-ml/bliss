import datetime
import sys
from pathlib import Path
from typing import TextIO

import pytorch_lightning as L
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader, Dataset

from bliss.datasets.saved_datasets import SavedGalsimBlends

NUM_WORKERS = 0


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
    log_file: TextIO = sys.stdout,
):
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_dl = DataLoader(val_ds, batch_size=batch_size, num_workers=num_workers)

    ccb = ModelCheckpoint(
        filename="epoch={epoch}-val_loss={val/loss:.3f}",
        save_top_k=5,
        verbose=True,
        monitor="val/loss",
        mode="min",
        save_on_train_epoch_end=False,
        auto_insert_metric_name=False,
    )

    logger = TensorBoardLogger(save_dir="out", name=model_name, default_hp_metric=False)
    print(f"INFO: Saving model as version {logger.version}", file=log_file)

    trainer = L.Trainer(
        limit_train_batches=1.0,
        max_epochs=n_epochs,
        logger=logger,
        callbacks=[ccb],
        accelerator="gpu",
        devices=1,
        log_every_n_steps=log_every_n_steps,
        check_val_every_n_epoch=validate_every_n_epoch,
        val_check_interval=val_check_interval,
    )

    return train_dl, val_dl, trainer


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
):
    assert model_name in {"detection", "binary", "deblender"}

    with open("log.txt", "a") as f:
        now = datetime.datetime.now()
        print("", file=f)
        log_msg = f"""Run training {model_name} encoder script...
        With seed {seed} at {now} validate_every_n_epoch {validate_every_n_epoch},
        val_check_interval {val_check_interval}, batch_size {batch_size}, n_epochs {n_epochs}

        Using datasets: {train_file}, {val_file}
        """
        print(log_msg, file=f)

    L.seed_everything(seed)

    if not Path(train_file).exists() and Path(val_file).exists():
        raise IOError("Training datasets do not exists")

    with open("log.txt", "a") as g:
        train_ds = SavedGalsimBlends(train_file)
        val_ds = SavedGalsimBlends(val_file)
        train_dl, val_dl, trainer = setup_training_objects(
            train_ds=train_ds,
            val_ds=val_ds,
            batch_size=batch_size,
            num_workers=NUM_WORKERS,
            n_epochs=n_epochs,
            validate_every_n_epoch=validate_every_n_epoch,
            val_check_interval=val_check_interval,
            model_name=model_name,
            log_every_n_steps=log_every_n_steps,
            log_file=g,
        )

    trainer.fit(model=model, train_dataloaders=train_dl, val_dataloaders=val_dl)
