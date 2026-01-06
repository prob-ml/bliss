#!/usr/bin/env python3

import subprocess

import pytorch_lightning as L
import typer

from experiment import DATASETS_DIR, MODELS_DIR, SEED, TORCH_DIR


def _save_model(*, model: str, version: int):
    ckpt_dir = TORCH_DIR / model / f"version_{version}" / "checkpoints"
    save_cmd = f"./scripts/get_model_from_checkpoint.py --model {model} --seed {SEED} --checkpoint-dir {ckpt_dir}"
    subprocess.check_call(save_cmd, shell=True)


def main(
    autoencoder: bool = False,
    detection: bool = False,
    binary: bool = False,
    deblend: bool = False,
    all: bool = False,
    version: int = 0,
    no_save: bool = False,
):
    L.seed_everything(SEED)

    # create paths to save models if they do not exist
    MODELS_DIR.mkdir(exist_ok=True)
    TORCH_DIR.mkdir(exist_ok=True)

    if autoencoder or all:
        train_ds_path = DATASETS_DIR / f"train_ae_ds_{SEED}.npz"
        val_ds_path = DATASETS_DIR / f"val_ae_ds_{SEED}.npz"
        cmd = f"./scripts/train/run_autoencoder_train.py --seed {SEED} --train-file {train_ds_path} --val-file {val_ds_path} --version {version}"
        subprocess.check_call(cmd, shell=True)

        if not no_save:
            _save_model(model="autoencoder", version=version)

    if detection or all:
        train_ds_path = DATASETS_DIR / f"train_ds_{SEED}.npz"
        val_ds_path = DATASETS_DIR / f"val_ds_{SEED}.npz"
        cmd = f"./scripts/train/run_detection_train.py --seed {SEED} --train-file {train_ds_path} --val-file {val_ds_path} --version {version}"
        subprocess.check_call(cmd, shell=True)

        if not no_save:
            _save_model(model="detection", version=version)

    if binary or all:
        train_ds_path = DATASETS_DIR / f"train_ds_{SEED}.npz"
        val_ds_path = DATASETS_DIR / f"val_ds_{SEED}.npz"
        cmd = f"./scripts/train/run_binary_train.py --seed {SEED} --train-file {train_ds_path} --val-file {val_ds_path} --version {version}"
        subprocess.check_call(cmd, shell=True)

        if not no_save:
            _save_model(model="binary", version=version)

    if deblend or all:
        train_ds_path = DATASETS_DIR / f"train_ds_deblend_{SEED}.npz"
        val_ds_path = DATASETS_DIR / f"val_ds_deblend_{SEED}.npz"
        ae_fpath = MODELS_DIR / f"autoencoder_{SEED}.pt"
        assert ae_fpath.exists()
        cmd = f"./scripts/train/run_deblender_train.py --seed {SEED} --ae-model-path {ae_fpath} --train-file {train_ds_path} --val-file {val_ds_path} --version {version}"
        subprocess.check_call(cmd, shell=True)
        if not no_save:
            _save_model(model="deblender", version=version)


if __name__ == "__main__":
    typer.run(main)
