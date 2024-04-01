#!/usr/bin/env python3

import matplotlib.pyplot as plt
import pytorch_lightning as L
import torch
from astropy.table import Table
from einops import rearrange
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from bliss.datasets.background import add_noise_and_background

# dataset
from bliss.datasets.galsim_blends import SavedGalsimBlends, generate_dataset, parse_dataset
from bliss.datasets.lsst import get_default_lsst_background, get_default_lsst_psf
from bliss.datasets.table_utils import column_to_tensor
from bliss.encoders.detection import DetectionEncoder
from bliss.encoders.layers import ConcatBackgroundTransform

OVERWRITE = True
N_SAMPLES = 1500
SPLIT = N_SAMPLES * 2 // 3
BATCH_SIZE = 32
NUM_WORKERS = 0
N_EPOCHS = 10
ONLY_BRIGHT = True

# device
gpu = torch.device("cuda:0")

# create datasets
if OVERWRITE:
    # prepare bigger dataset
    catsim_table = Table.read("../../../data/OneDegSq.fits")
    all_star_mags = column_to_tensor(Table.read("../../../data/stars_med_june2018.fits"), "i_ab")
    psf = get_default_lsst_psf()

    if ONLY_BRIGHT:
        mask = (catsim_table["i_ab"] > 22) & (catsim_table["i_ab"] < 23)
        catsim_table = catsim_table[mask]
        print("INFO: Smaller catalog with only bright sources of length:", len(catsim_table))

    dataset = generate_dataset(N_SAMPLES, catsim_table, all_star_mags, 4, 10, psf)

    # train, test split
    train_ds = {p: q[:SPLIT] for p, q in dataset.items()}
    val_ds = {p: q[SPLIT:] for p, q in dataset.items()}

    # now save  data
    torch.save(train_ds, "train_ds.pt")
    torch.save(val_ds, "val_ds.pt")

train_dataset = SavedGalsimBlends("train_ds.pt", SPLIT)
validation_dataset = SavedGalsimBlends("val_ds.pt", N_SAMPLES - SPLIT)


# now dataloaders

train_dl = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=NUM_WORKERS
)
val_dl = DataLoader(
    validation_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=True
)

# callback
checkpoint_callback = ModelCheckpoint(
    filename="epoch={epoch}-val_loss={val/loss:.3f}",
    save_top_k=1,
    verbose=True,
    monitor="val/loss",
    mode="min",
    save_on_train_epoch_end=False,
    auto_insert_metric_name=False,
)

# logger
logger = TensorBoardLogger(
    save_dir="out",
    name="detection",
    default_hp_metric=False,
)

# now train on the same batch 100 times with some optimizer
input_transform = ConcatBackgroundTransform()
detection_encoder = DetectionEncoder(input_transform)


trainer = L.Trainer(
    limit_train_batches=1.0,
    max_epochs=1000,
    logger=logger,
    callbacks=[checkpoint_callback],
    accelerator="gpu",
    devices=1,
    log_every_n_steps=16,
    check_val_every_n_epoch=10,
)
trainer.fit(model=detection_encoder, train_dataloaders=train_dl, val_dataloaders=val_dl)
