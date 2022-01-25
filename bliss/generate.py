import math
import os
from pathlib import Path

import torch
from hydra.utils import instantiate
from matplotlib import pyplot as plt
from omegaconf import DictConfig, OmegaConf

from bliss.reporting import plot_image


def visualize(batch, path, n_samples, figsize=(12, 12)):
    # visualize in a pdf format 30 images from the batch
    assert math.sqrt(n_samples) % 1 == 0
    nrows = int(math.sqrt(n_samples))

    fig, axes = plt.subplots(nrows=nrows, ncols=nrows, figsize=figsize)
    axes = axes.flatten()
    images = batch["images"]
    assert len(images.shape) == 4
    for i in range(n_samples):
        # get first band of image in numpy format.
        ax = axes[i]
        image = images[i][0].cpu().numpy()
        plot_image(fig, ax, image)

    plt.tight_layout()
    fig.savefig(path, bbox_inches="tight")


def generate(cfg: DictConfig):
    # setup
    paths = OmegaConf.to_container(cfg.paths, resolve=True)
    output = Path(paths["output"])
    if not os.path.exists(output.as_posix()):
        os.makedirs(output.as_posix())

    filepath = Path(cfg.generate.file)
    imagepath = Path(filepath.parent).joinpath(filepath.stem + "_images.pdf")
    dataset = instantiate(cfg.generate.dataset)

    # params common to all batches (do not stack).
    global_params = set(cfg.generate.common)

    # get batches and combine them
    fbatch = {}
    for batch in dataset.train_dataloader():
        if not bool(fbatch):  # dict is empty
            fbatch = batch
            for key, val in fbatch.items():
                if key in global_params:
                    fbatch[key] = val[0]
        else:
            for key, val in fbatch.items():
                if key not in global_params:
                    fbatch[key] = torch.vstack((val, batch[key]))

    # make sure in CPU by default.
    # assumes all data are tensors (including metadata).
    fbatch = {k: v.cpu() for k, v in fbatch.items()}

    # save batch and images as pdf for visualization purposes.
    torch.save(fbatch, filepath)
    visualize(fbatch, imagepath, cfg.generate.n_plots)
