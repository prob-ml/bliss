import math
from typing import Any, Dict

import torch
import tqdm
from matplotlib import pyplot as plt

from bliss.plotting import plot_image


def visualize(batch, path, n_samples, figsize=(12, 12)):
    # visualize 30 images from the batch
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


def generate(dataset, filepath, imagepath, n_plots: int, global_params=("background", "slen")):
    # setup

    # params common to all batches (do not stack).
    global_params = set(global_params)

    # get batches and combine them
    fbatch: Dict[str, Any] = {}
    for batch in tqdm.tqdm(dataset.train_dataloader(), desc="Generating dataset"):
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

    # save batch and images for visualization purposes.
    torch.save(fbatch, filepath)
    visualize(fbatch, imagepath, n_plots)
