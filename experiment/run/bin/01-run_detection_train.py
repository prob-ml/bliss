#!/usr/bin/env python3

from pathlib import Path

import pytorch_lightning as L

from bliss.encoders.detection import DetectionEncoder
from experiment.run.training_functions import create_dataset, setup_training_objects

OVERWRITE = True
N_SAMPLES = 1028 * 20
SPLIT = N_SAMPLES * 15 // 20
BATCH_SIZE = 32
NUM_WORKERS = 0
N_EPOCHS = 25
ONLY_BRIGHT = False
VERSION = "10_1"  # for dataset
TRAIN_DS_FILE = f"ds/train_ds_{VERSION}.pt"
VAL_DS_FILE = f"ds/val_ds_{VERSION}.pt"
VALIDATE_EVERY_N_EPOCH = 1
VAL_CHECK_INTERVAL = 32
ADD_PADDING_GALAXIES = True
STAR_DENSITY = 0
GALAXY_DENSITY = 185
SEED = 43

L.seed_everything(SEED)


# setup model to train
detection_encoder = DetectionEncoder()


if OVERWRITE:
    create_dataset(
        "../../../data/OneDegSq.fits",
        "../../../data/stars_med_june2018.fits",
        N_SAMPLES,
        SPLIT,
        TRAIN_DS_FILE,
        VAL_DS_FILE,
        only_bright=ONLY_BRIGHT,
        add_galaxies_in_padding=ADD_PADDING_GALAXIES,
        galaxy_density=GALAXY_DENSITY,
        star_density=STAR_DENSITY,
    )

if not OVERWRITE and not Path(TRAIN_DS_FILE).exists():
    raise IOError("No file found.")

train_dl, val_dl, trainer = setup_training_objects(
    TRAIN_DS_FILE,
    VAL_DS_FILE,
    N_SAMPLES,
    SPLIT,
    BATCH_SIZE,
    NUM_WORKERS,
    N_EPOCHS,
    VALIDATE_EVERY_N_EPOCH,
    VAL_CHECK_INTERVAL,
)

trainer.fit(model=detection_encoder, train_dataloaders=train_dl, val_dataloaders=val_dl)
