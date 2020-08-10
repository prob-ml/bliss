import matplotlib.pyplot as plt
import torch
from mpl_toolkits.axes_grid1 import make_axes_locatable


def plot_locs(ax, slen, locs, color="r", marker="x", s=1):
    assert len(locs.shape) == 2
    assert locs.shape[1] == 2
    ax.scatter(
        x=locs[:, 1] * (slen - 1),
        y=locs[:, 0] * (slen - 1),
        color=color,
        marker=marker,
        s=s,
    )


def plot_image(
    fig, ax, image, true_locs=None, estimated_locs=None, vmin=None, vmax=None, s=5,
):

    # locations are coordinates in the image, on scale from 0 to 1

    slen = image.shape[-1]

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    im = ax.matshow(image, vmin=vmin, vmax=vmax)

    if true_locs:
        plot_locs(ax, slen, true_locs, color="r", marker="x", s=s)

    if estimated_locs:
        plot_locs(ax, slen, estimated_locs, color="b", marker="o", s=s)

    fig.colorbar(im, cax=cax, orientation="vertical")
