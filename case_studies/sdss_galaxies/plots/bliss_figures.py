"""Produce all figures. Save to PNG format."""
from abc import abstractmethod
from pathlib import Path

import matplotlib as mpl
import numpy as np
import pytorch_lightning as pl
import seaborn as sns
import torch
from matplotlib import pyplot as plt
from torch import Tensor

pl.seed_everything(40)


CB_color_cycle = [
    "#377eb8",
    "#ff7f00",
    "#4daf4a",
    "#f781bf",
    "#a65628",
    "#984ea3",
    "#999999",
    "#e41a1c",
    "#dede00",
]


def set_rc_params(
    figsize=(10, 10),
    fontsize=18,
    title_size="large",
    label_size="medium",
    legend_fontsize="medium",
    tick_label_size="small",
    major_tick_size=7,
    minor_tick_size=4,
    major_tick_width=0.8,
    minor_tick_width=0.6,
    lines_marker_size=8,
):
    # named size options: 'xx-small', 'x-small', 'small', 'medium', 'large', 'x-large', 'xx-large'.
    rc_params = {
        # font.
        "font.family": "serif",
        "font.sans-serif": "Helvetica",
        "text.usetex": True,
        "text.latex.preamble": r"\usepackage{amsmath}",
        "mathtext.fontset": "cm",
        "font.size": fontsize,
        # figure
        "figure.figsize": figsize,
        # axes
        "axes.labelsize": label_size,
        "axes.titlesize": title_size,
        # ticks
        "xtick.labelsize": tick_label_size,
        "ytick.labelsize": tick_label_size,
        "xtick.major.size": major_tick_size,
        "ytick.major.size": major_tick_size,
        "xtick.major.width": major_tick_width,
        "ytick.major.width": major_tick_width,
        "ytick.minor.size": minor_tick_size,
        "xtick.minor.size": minor_tick_size,
        "xtick.minor.width": minor_tick_width,
        "ytick.minor.width": minor_tick_width,
        # markers
        "lines.markersize": lines_marker_size,
        # legend
        "legend.fontsize": legend_fontsize,
        # colors
        "axes.prop_cycle": mpl.cycler(color=CB_color_cycle),
        # images
        "image.cmap": "gray",
    }
    mpl.rcParams.update(rc_params)
    sns.set_context(rc=rc_params)


def format_plot(ax, xlims=None, ylims=None, xticks=None, yticks=None, xlabel="", ylabel=""):
    if xlims is not None:
        ax.set_xlim(xlims)
    if ylims is not None:
        ax.set_ylim(ylims)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if xticks is not None:
        ax.set_xticks(xticks)
    if yticks is not None:
        ax.set_yticks(yticks)


def to_numpy(d: dict):
    for k, v in d.items():
        if isinstance(v, Tensor):
            d[k] = v.numpy()
        elif isinstance(v, (float, int, np.ndarray)):
            d[k] = v
        elif isinstance(v, dict):
            v = to_numpy(v)
            d[k] = v
        else:
            msg = f"Data returned can only be dict, tensor, array, or float but got {type(v)}"
            raise TypeError(msg)
    return d


class BlissFigures:
    cache = "temp.pt"

    def __init__(self, figdir, cachedir, overwrite=False, img_format="png") -> None:

        self.figdir = Path(figdir)
        self.cachefile = Path(cachedir) / self.cache
        self.overwrite = overwrite
        self.img_format = img_format

    def get_data(self, *args, **kwargs):
        """Return summary of data for producing plot, must be cachable w/ torch.save()."""
        if self.cachefile.exists() and not self.overwrite:
            return torch.load(self.cachefile)

        data = self.compute_data(*args, **kwargs)
        torch.save(data, self.cachefile)
        return data

    @abstractmethod
    def compute_data(self, *args, **kwargs) -> dict:
        """Should only return tensors that can be casted to numpy."""
        return {}

    def save_figures(self, *args, **kwargs):
        """Create figures and save to output directory with names from `self.fignames`."""
        data = self.get_data(*args, **kwargs)
        data_np = to_numpy(data)
        figs = self.create_figures(data_np)  # data for figures is all numpy arrays or floats.
        for figname, fig in figs.items():
            figfile = self.figdir / f"{figname}.{self.img_format}"
            fig.savefig(figfile, format=self.img_format)
            plt.close(fig)

    @abstractmethod
    def create_figures(self, data):
        """Return matplotlib figure instances to save based on data."""
        return {"temp_fig": mpl.figure.Figure()}
