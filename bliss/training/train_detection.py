#!/usr/bin/env python3

from pathlib import Path

import click
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from bliss.datasets.galsim_blends import SavedGalsimBlends, parse_dataset
from bliss.encoders.detection import DetectionEncoder
from bliss.encoders.layers import ConcatBackgroundTransform

HOME_DIR = Path(__file__).parent.parent.parent
DATASET_DIR = HOME_DIR / "case_studies/galsim_galaxies/data"
LOG_DIR = HOME_DIR / "case_studies/galsim_galaxies/output/detection_encoder"


def train_one_epoch(
    epoch_indx: int,
    model: DetectionEncoder,
    training_loader: DataLoader,
    optimizer: Adam,
    writer: SummaryWriter | None = None,
    log_every_n_batches: int = 50,
) -> float:

    model.train()  # need to set in train mode
    device = model.device
    running_loss = 0
    running_counter_loss = 0
    running_locs_loss = 0
    last_loss = 0

    for ii, batch in enumerate(training_loader):

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # get lost and backward propagate
        images, background, true_cat = parse_dataset(batch)

        # make sure correct device is being used
        images.to(device)
        background.to(device)
        true_cat.to(device)

        losses = model.get_loss(images, background, true_cat)
        loss = losses["loss"]
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()
        running_counter_loss += losses["counter_loss"]
        running_locs_loss += losses["locs_loss"]

        if ii % log_every_n_batches == log_every_n_batches - 1 and writer:

            # loss per batch
            last_loss = running_loss / log_every_n_batches
            last_counter_loss = running_counter_loss / log_every_n_batches
            last_locs_loss = running_locs_loss / log_every_n_batches

            tb_x = epoch_indx * len(training_loader) + ii + 1
            writer.add_scalar("train/loss", last_loss, tb_x)
            writer.add_scalar("train/counter_loss", last_counter_loss, tb_x)
            writer.add_scalar("train/locs_loss", last_locs_loss, tb_x)
            writer.flush()
            running_loss = 0

    return last_loss


def validate(
    epoch: int,
    model: nn.Module,
    val_loader: DataLoader,
    writer: SummaryWriter | None = None,
    model_path: Path | None = None,
    best_vloss: float | None = None,
) -> float:

    # need to set model in eval mode
    model.eval()
    device = model.device

    running_vloss = 0
    running_v_counter_loss = 0
    running_v_locs_loss = 0
    val_n_batches = 0

    # Disable gradient computation and reduce memory consumption.
    with torch.no_grad():
        for batch in val_loader:
            images, background, true_cat = parse_dataset(batch)

            # make sure correct device is being used
            images = images.to(device)
            images = background.to(device)
            true_cat = true_cat.to(device)

            losses = model.get_loss(images, background, true_cat)
            running_vloss += losses["loss"].detach().item()
            running_v_counter_loss += losses["counter_loss"]
            running_v_locs_loss += losses["locs_loss"]
            val_n_batches += 1

    avg_vloss = running_vloss / (val_n_batches + 1)
    avg_v_counter_loss = running_v_counter_loss / (val_n_batches + 1)
    avg_v_locs_loss = running_v_locs_loss / (val_n_batches + 1)

    if writer:
        writer.add_scalars("val/loss", avg_vloss, epoch + 1)
        writer.add_scalars("val/counter_loss", avg_v_counter_loss, epoch + 1)
        writer.add_scalars("val/locs_loss", avg_v_locs_loss, epoch + 1)
        writer.flush()

    # Track best performance, and save the model's state
    if model_path and best_vloss and avg_vloss < best_vloss:
        best_vloss = avg_vloss
        path = model_path.parent / f"{model_path.stem}_{epoch}{model_path.suffix}"
        torch.save(model.state_dict(), path)

    return best_vloss


@click.command()
@click.option("-v", "--version", default=1)
@click.option("-e", "--n-epochs", default=1)
@click.option("--batch-size", default=32)
@click.option("--train-ds-file", default="train_gbs.pt")
@click.option("--val-ds-file", default="val_gbs.pt")
def main(version: int, n_epochs: int, batch_size: int, train_ds_file: str, val_ds_file: str):

    target_dir = LOG_DIR / f"v{version}"
    if not target_dir.exists():
        target_dir.mkdir()
    model_path = target_dir / "detection.pt"

    train_ds = SavedGalsimBlends(train_ds_file, 1000)
    val_ds = SavedGalsimBlends(val_ds_file, 1000)

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size)

    input_transform = ConcatBackgroundTransform()
    detection_encoder = DetectionEncoder(input_transform)

    opt = Adam(detection_encoder.parameters(), lr=1e-4)

    writer = SummaryWriter(log_dir=target_dir)

    best_vloss = torch.inf

    for epoch_indx in range(n_epochs):
        train_one_epoch(
            epoch_indx, detection_encoder, train_dl, opt, writer, log_every_n_batches=10
        )

        validate(epoch_indx, detection_encoder, val_dl, writer, model_path, best_vloss)


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
