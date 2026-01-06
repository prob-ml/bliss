#!/usr/bin/env python3

from pathlib import Path

import numpy as np
import torch
import typer

from experiment import MODELS_DIR


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


def main(
    model: str = typer.Option(),
    seed: int = typer.Option(),
    checkpoint_dir: str = typer.Option(),
):
    """Save weights from model checkpoint."""
    weight_path = MODELS_DIR / f"{model}_{seed}.pt"
    checkpoint_path = _find_best_checkpoint(checkpoint_dir)
    _save_weights(weight_path, checkpoint_path)


if __name__ == "__main__":
    typer.run(main)
