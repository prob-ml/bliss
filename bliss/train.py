from pathlib import Path
from typing import Callable

import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


def train_one_epoch(
    epoch_indx: int,
    model: nn.Module,
    training_loader: DataLoader,
    loss_fn: Callable,
    optimizer: Adam,
    writer: SummaryWriter | None = None,
    log_every_n_batches: int = 1000,
) -> float:

    running_loss = 0
    last_loss = 0

    for ii, data in enumerate(training_loader):

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        outputs = model(*data)

        # Compute the loss and its gradients
        loss = loss_fn(*data, *outputs)
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()
        if ii % log_every_n_batches == log_every_n_batches - 1 and writer:
            last_loss = running_loss / log_every_n_batches  # loss per batch
            tb_x = epoch_indx * len(training_loader) + ii + 1
            writer.add_scalar("train/loss", last_loss, tb_x)
            running_loss = 0

    return last_loss


def validate(
    epoch: int,
    model: nn.Module,
    val_loader: DataLoader,
    loss_fn: Callable,
    writer: SummaryWriter | None = None,
    model_path: Path | None = None,
    best_vloss: float | None = None,
) -> float:
    model.eval()

    running_vloss = 0
    val_n_batches = 0

    # Disable gradient computation and reduce memory consumption.
    with torch.no_grad():
        for vdata in enumerate(val_loader):
            vinputs, vlabels = vdata
            voutputs = model(vinputs)
            vloss = loss_fn(voutputs, vlabels)
            running_vloss += vloss
            val_n_batches += 1

    avg_vloss = running_vloss / (val_n_batches + 1)

    if writer:
        writer.add_scalars("val/loss", avg_vloss, epoch + 1)
        writer.flush()

    # Track best performance, and save the model's state
    if model_path and best_vloss and avg_vloss < best_vloss:
        best_vloss = avg_vloss
        path = model_path.parent / f"{model_path.stem}_{epoch}{model_path.suffix}"
        torch.save(model.state_dict(), path)

    return best_vloss


def train(
    n_epochs: int,
    model: nn.Module,
    training_loader: DataLoader,
    loss_fn: Callable,
    optimizer: Adam,
    log_every_n_batches: int = 1000,
    val_loader: DataLoader | None = None,
    model_path: str | Path | None = None,
    val_every_n_epoch: int = 1,
    writer: SummaryWriter | None = None,
):

    best_vloss = torch.inf

    for epoch in tqdm(n_epochs, desc="N_EPOCHS:"):

        model.train(True)
        train_one_epoch(
            epoch,
            model,
            training_loader,
            loss_fn,
            optimizer,
            writer,
            log_every_n_batches=log_every_n_batches,
        )

        if val_loader and epoch % val_every_n_epoch == val_every_n_epoch - 1:
            best_vloss = validate(epoch, model, val_loader, loss_fn, writer, model_path, best_vloss)
