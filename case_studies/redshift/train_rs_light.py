# Author: Qiaozhi Huang
# train to predict redshift using network and pytorch lightning framework

import os
import re

import click
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader, TensorDataset

from case_studies.redshift.network_rs import PhotoZFromFluxes


@click.command()
@click.option("--train_path", help="Train path", required=True, type=str)
@click.option("--val_path", help="Test path", required=True, type=str)
@click.option("--outdir", help="Output directory", metavar="[PATH|URL]", required=True, type=str)
# Optinal features
@click.option("--epoch", help="Num of epoch to train", default=5000, type=int)
@click.option("--resume", help="Resume from given network", metavar="[PATH|URL]", type=str)
@click.option("--nick", help="Nickname for remember", type=str)
def main(epoch, train_path, val_path, nick, resume, outdir):  # noqa: WPS216
    ###### Prepare Training set  # noqa: E266
    print("start reading dataset!")  # noqa: WPS421

    path_train = train_path
    path_val = val_path
    photo_z_train = pd.read_pickle(path_train)
    photo_z_val = pd.read_pickle(path_val)
    print("finish reading dataset!")  # noqa: WPS421

    # options for dataset
    dataset_options = {
        "batch_size": 512,
        "num_bins": 1,
        "group_size": 128,
        "x_dim": 6,
        "y_dim": 1,
    }

    print("start tensor dataset preparation!")  # noqa: WPS421
    dataloader_train = preprocess(
        photo_z_train,
        dataset_options["batch_size"],
        dataset_options["group_size"],
        dataset_options["num_bins"],
        dataset_options["x_dim"],
        dataset_options["y_dim"],
    )
    dataloader_val = preprocess(
        photo_z_val,
        dataset_options["batch_size"],
        dataset_options["group_size"],
        dataset_options["num_bins"],
        dataset_options["x_dim"],
        dataset_options["y_dim"],
    )
    print("finish tensor dataset preparation!")  # noqa: WPS421

    ###### Construct Network # noqa: E266
    # options for network
    network_options = {
        "hidden_dim": 256,
        "out_dim": dataset_options["num_bins"],
        "n_epochs": epoch,
        "outdir": outdir,
        "snap": 1,  # how many epoches to save one model once
        "loss_fcn": torch.nn.MSELoss()
        if dataset_options["num_bins"] == 1
        else torch.nn.CrossEntropyLoss(),  # loss func
        "dropout_rate": 0.5,
        "learning_rate": 1e-3,
        "group_size": dataset_options["group_size"],
        "num_gpu": 1,
        "gpu_device": [0],
        "log_every_n_steps": 1000,  # How often to log within steps
        "val_check_interval": 0.2,  # How often to check the validation set.
    }
    in_dim = dataset_options["x_dim"]
    reg = PhotoZFromFluxes(
        in_dim,
        network_options["hidden_dim"],
        network_options["out_dim"],
        network_options["dropout_rate"],
        network_options["learning_rate"],
        network_options["loss_fcn"],
    )

    ###### Start Train # noqa: E266
    # Pick output directory.
    outdir = network_options["outdir"]
    os.makedirs(outdir, exist_ok=True)
    prev_run_dirs = []
    if os.path.isdir(outdir):
        prev_run_dirs = [x for x in os.listdir(outdir) if os.path.isdir(os.path.join(outdir, x))]
    prev_run_ids = [re.match(r"^\d+", x) for x in prev_run_dirs]
    prev_run_ids = [int(x.group()) for x in prev_run_ids if x is not None]
    cur_run_id = max(prev_run_ids, default=-1) + 1
    nickname = ""
    if nick is not None:
        nickname = nick
    run_dir = os.path.join(outdir, f"{cur_run_id:05d}-run{nickname}")
    os.makedirs(run_dir)
    logger = TensorBoardLogger(save_dir=run_dir, name="tensorboard_logs")

    # train
    print("start training")  # noqa: WPS421
    if resume is not None:
        reg = reg.load_from_checkpoint(resume)
        # Set the model to training mode
        reg.train()

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        filename="reg_{val_loss:.6f}_{epoch:02d}",
        every_n_epochs=network_options["snap"],
        save_top_k=5,
    )
    trainer = pl.Trainer(
        max_epochs=network_options["n_epochs"],
        default_root_dir=run_dir,
        logger=logger,
        accelerator="gpu",
        devices=network_options["gpu_device"],
        callbacks=[checkpoint_callback],
        log_every_n_steps=network_options["log_every_n_steps"],
        val_check_interval=network_options["val_check_interval"],
    )
    trainer.fit(model=reg, train_dataloaders=dataloader_train, val_dataloaders=dataloader_val)
    print("finish training!")  # noqa: WPS421


def preprocess(
    df: pd.DataFrame,
    batch_size: int,
    group_size: int = 1,
    num_bins: int = 1,
    x_dim: int = 6,
    y_dim: int = 1,
) -> DataLoader:
    """Preprocess dataframe to dataloader.

    Args:
        df: Dataframe
        batch_size: int
        group_size: int size for each group
        num_bins: int num of bins
        x_dim: int dim of input value
        y_dim: int dim of predicted value

    Returns:
        dataloader
    """
    x = df.values[:, :x_dim].astype(float)
    y = df.values[:, -y_dim].astype(float)
    n_samples, n_features_x = x.shape
    n_features_y = y_dim
    n_samples = n_samples // group_size * group_size

    x_train = np.array(x[:n_samples])
    y_train = np.array(y[:n_samples])
    if num_bins != 1:
        y_train = bin_target(y_train, num_bins)  # bin
    if group_size != 1:
        tensor_x = torch.Tensor(x_train).view(-1, group_size, n_features_x)
        tensor_y = torch.Tensor(y_train).view(-1, group_size, n_features_y)
    else:
        tensor_x = torch.Tensor(x_train)
        tensor_y = torch.Tensor(y_train)
    custom_dataset = TensorDataset(tensor_x, tensor_y)
    return DataLoader(custom_dataset, batch_size=batch_size, pin_memory=True)


def bin_target(y: np.array, num_bins: int = 30):
    """Make numpy array into bin values.

    Args:
        y: ndarray
        num_bins: int

    Returns:
        ndarry representing the respective bin(indices) for each value of y
    """
    bin_edges = np.linspace(y.min(), y.max(), num_bins + 1)
    return pd.cut(y, bins=bin_edges, labels=False, include_lowest=True)


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
