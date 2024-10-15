#!/usr/bin/env python3

import datetime
from pathlib import Path

import click
import numpy as np
import torch

from bliss import HOME_DIR

MODELS_DIR = HOME_DIR / "experiment/models"
LOG_FILE = HOME_DIR / "experiment/log.txt"


def _find_best_checkpoint(checkpoint_dir: str):
    """Given directory to checkpoints, automatically return file path to lowest loss checkpoint."""
    best_path = Path(".")
    min_loss = np.inf
    for pth in Path(checkpoint_dir).iterdir():
        if pth.stem.startswith("epoch"):
            # extract loss
            idx = pth.stem.find("=", len("epoch") + 1)
            loss = float(pth.stem[idx + 1 :])
            if loss < min_loss:
                best_path = pth
                min_loss = loss
    return best_path


def _save_weights(weight_save_path: str, model_checkpoint_path: str):
    model_checkpoint = torch.load(model_checkpoint_path, map_location="cpu", weights_only=True)
    model_state_dict = model_checkpoint["state_dict"]
    weight_file_path = Path(weight_save_path)
    assert weight_file_path.parent.exists()
    assert not weight_file_path.exists(), "Weight with same seed already exists."
    torch.save(model_state_dict, weight_save_path)


@click.command()
@click.option("-m", "--model", type=str, required=True)
@click.option("--seed", type=int, required=True)  # used to train model
@click.option("--ds-seed", type=int, required=True)  # used to produce dataset
@click.option("-c", "--checkpoint-dir", type=str, required=True)
def main(model: str, seed: int, ds_seed: int, checkpoint_dir: str):
    """Save weights from model checkpoint."""
    weight_path = MODELS_DIR / f"{model}_{ds_seed}_{seed}.pt"

    checkpoint_path = _find_best_checkpoint(checkpoint_dir)
    _save_weights(weight_path, checkpoint_path)

    with open(LOG_FILE, "a", encoding="utf-8") as f:
        now = datetime.datetime.now()
        print(
            f"\n Saved checkpoint '{checkpoint_path}' as weights {weight_path} at {now}",
            file=f,
        )


if __name__ == "__main__":
    main()
