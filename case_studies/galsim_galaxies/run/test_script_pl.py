#!/usr/bin/env python3

from pathlib import Path

import pytorch_lightning as L
import torch
from astropy.table import Table
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

from bliss.datasets.galsim_blends import SavedGalsimBlends, generate_dataset
from bliss.datasets.lsst import get_default_lsst_psf
from bliss.datasets.table_utils import column_to_tensor
from bliss.encoders.detection import DetectionEncoder
from bliss.encoders.layers import ConcatBackgroundTransform

OVERWRITE = True
N_SAMPLES = 1028 * 20
SPLIT = N_SAMPLES * 15 // 20
BATCH_SIZE = 32
NUM_WORKERS = 0
N_EPOCHS = 100
ONLY_BRIGHT = False
VERSION = "4"  # for dataset
TRAIN_DATASET_FILE = f"train_ds_{VERSION}.pt"
VALIDATION_DATASET_FILE = f"val_ds_{VERSION}.pt"
VALIDATE_EVERY_N_EPOCH = 1
VAL_CHECK_INTERVAL = 32

# device
gpu = torch.device("cuda:0")

if not OVERWRITE and not Path(TRAIN_DATASET_FILE).exists():
    raise IOError

# create datasets
if OVERWRITE:
    print("INFO: Overwriting dataset...")
    # prepare bigger dataset
    catsim_table = Table.read("../../../data/OneDegSq.fits")
    all_star_mags = column_to_tensor(Table.read("../../../data/stars_med_june2018.fits"), "i_ab")
    psf = get_default_lsst_psf()

    if ONLY_BRIGHT:
        bright_mask = catsim_table["i_ab"] < 23
        new_table = catsim_table[bright_mask]
        print("INFO: Smaller catalog with only bright sources of length:", len(new_table))

    else:
        mask = catsim_table["i_ab"] < 27.3
        new_table = catsim_table[mask]
        print("INFO: Complete catalog with only i < 27.3 magnitude of length:", len(new_table))

    dataset = generate_dataset(
        N_SAMPLES,
        new_table,
        all_star_mags,
        mean_sources=4,
        max_n_sources=10,
        psf=psf,
        slen=40,
        bp=24,
        max_shift=0.5,
        galaxy_prob=1.0,
    )

    # train, test split
    train_ds = {p: q[:SPLIT] for p, q in dataset.items()}
    val_ds = {p: q[SPLIT:] for p, q in dataset.items()}

    # now save  data
    torch.save(train_ds, TRAIN_DATASET_FILE)
    torch.save(val_ds, VALIDATION_DATASET_FILE)

train_dataset = SavedGalsimBlends(TRAIN_DATASET_FILE, SPLIT)
validation_dataset = SavedGalsimBlends(VALIDATION_DATASET_FILE, N_SAMPLES - SPLIT)


# now dataloaders
train_dl = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
val_dl = DataLoader(validation_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

# callback
checkpoint_callback = ModelCheckpoint(
    filename="epoch={epoch}-val_loss={val/loss:.3f}",
    save_top_k=5,
    verbose=True,
    monitor="val/loss",
    mode="min",
    save_on_train_epoch_end=False,
    auto_insert_metric_name=False,
)

# logger
logger = TensorBoardLogger(save_dir="out", name="detection", default_hp_metric=False)

# now train on the same batch 100 times with some optimizer
input_transform = ConcatBackgroundTransform()
detection_encoder = DetectionEncoder(input_transform)


trainer = L.Trainer(
    limit_train_batches=1.0,
    max_epochs=N_EPOCHS,
    logger=logger,
    callbacks=[checkpoint_callback],
    accelerator="gpu",
    devices=1,
    log_every_n_steps=16,
    check_val_every_n_epoch=VALIDATE_EVERY_N_EPOCH,
    val_check_interval=VAL_CHECK_INTERVAL,
)
trainer.fit(model=detection_encoder, train_dataloaders=train_dl, val_dataloaders=val_dl)
