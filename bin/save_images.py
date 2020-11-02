#!/usr/bin/env python3

import hydra
import torch
import matplotlib.pyplot as plt
from omegaconf import DictConfig
from pathlib import Path

from bliss.datasets import simulated
from bliss import plotting

datasets = {"SimulatedDataset": simulated.SimulatedDataset}


def visualize(batch, path, n_samples=25, nrows=5, figsize=(12, 12)):
    # visualize in a pdf format 30 images from the batch
    assert n_samples % nrows == 0
    assert nrows ** 2 == n_samples

    fig, axes = plt.subplots(nrows=nrows, ncols=nrows, figsize=figsize)
    axes = axes.flatten()
    images = batch["images"]
    assert len(images.shape) == 4
    for i in range(n_samples):
        # get first band of image in numpy format.
        ax = axes[i]
        image = images[i][0].cpu().numpy()
        plotting.plot_image(fig, ax, image)

    fig.savefig(path, bbox_inches="tight")


@hydra.main(config_path="../config", config_name="config")
def main(cfg: DictConfig):
    # setup
    filepath = Path(cfg.saving.file)
    imagepath = filepath.parent.joinpath(filepath.stem + "_images").with_suffix(".pdf")
    dataset = datasets[cfg.dataset.name](cfg)
    batch = dataset.get_batch()

    # save batch and images as pdf
    torch.save(batch, filepath)
    visualize(batch, imagepath)


if __name__ == "__main__":
    main()
