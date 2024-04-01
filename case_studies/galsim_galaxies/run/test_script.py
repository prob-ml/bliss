#!/usr/bin/env python3

import matplotlib.pyplot as plt
import torch
from astropy.table import Table
from einops import rearrange
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

OVERWRITE = False
N_SAMPLES = 999
SPLIT = N_SAMPLES * 2 // 3
BATCH_SIZE = 32
NUM_WORKERS = 0
N_EPOCHS = 10

# device
gpu = torch.device("cuda:0")

# create datasets
if OVERWRITE:
    # prepare bigger dataset
    catsim_table = Table.read("../../../data/OneDegSq.fits")
    all_star_mags = column_to_tensor(Table.read("../../../data/stars_med_june2018.fits"), "i_ab")
    psf = get_default_lsst_psf()

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


# now train on the same batch 100 times with some optimizer

input_transform = ConcatBackgroundTransform()
detection_encoder = DetectionEncoder(input_transform)

# gpu
detection_encoder.to(gpu)

opt = Adam(detection_encoder.parameters(), lr=1e-4)


for ii in tqdm(range(N_EPOCHS), desc="epoch:"):

    # train
    running_loss = 0.0
    running_locs_loss = 0.0
    running_counter_loss = 0.0
    train_n_batches = 0
    detection_encoder.train()
    for tbatch in train_dl:
        opt.zero_grad()
        images, background, truth_cat = parse_dataset(tbatch)
        images = images.to(gpu)
        background = background.to(gpu)
        truth_cat = truth_cat.to(gpu)
        losses = detection_encoder.get_loss(images, background, truth_cat)
        loss = losses["loss"]
        loss.backward()
        opt.step()

        running_loss += loss.detach().cpu().item()
        running_counter_loss += losses["counter_loss"]
        running_locs_loss += losses["locs_loss"]
        train_n_batches += 1

    running_loss /= train_n_batches
    running_counter_loss /= train_n_batches
    running_locs_loss /= train_n_batches

    print("epoch:", ii, ",training_loss: ", running_loss)
    print("epoch:", ii, ",counter_training_loss: ", running_counter_loss)
    print("epoch:", ii, ",counter_locs_loss: ", running_locs_loss)

    # if ii % 4 == 3:
    #     val_running_loss = 0.0
    #     val_n_batches = 0
    #     detection_encoder.eval()
    #     with torch.no_grad():
    #         for bval in val_dl:
    #             images, background, truth_catalog = parse_dataset(bval)
    #             images = images.to(gpu)
    #             background = background.to(gpu)
    #             truth_catalog = truth_catalog.to(gpu)

    #             val_loss = detection_encoder.get_loss(images, background, truth_catalog)["loss"]
    #             val_running_loss += val_loss.detach().cpu().item()
    #             val_n_batches += 1
    #     val_running_loss /= val_n_batches
    #     print("epoch: ", ii, ", val_loss: ", val_running_loss)
