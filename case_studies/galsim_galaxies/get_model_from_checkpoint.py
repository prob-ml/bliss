#!/usr/bin/env python3

from pathlib import Path

import click
import torch


def _save_best_weights(weight_save_path: str, model_checkpoint_path: str):
    model_checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    model_state_dict = model_checkpoint["state_dict"]
    weight_file_path = Path(weight_save_path)
    assert weight_file_path.parent.exists()
    torch.save(model_state_dict, weight_save_path)


@click.command()
@click.option("-w", "--weight-path", type=str, required=True)
@click.option("-c", "--checkpoint-path", type=str, required=True)
def main(weight_path: str, checkpoint_path: str):
    """Save weights from model checkpoint."""
    _save_best_weights(weight_path, checkpoint_path)

    with open("run/log.txt", "a", encoding="utf-8") as f:
        assert Path("run/log.txt").exists()
        print()
        print(f"INFO: Saved checkpoint '{checkpoint_path}' as weights {weight_path}", file=f)


if __name__ == "__main__":
    main()
