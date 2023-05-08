"""Common functions to plot results."""
from abc import abstractmethod
from pathlib import Path

import matplotlib as mpl
import numpy as np
import seaborn as sns
import torch
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from mpl_toolkits.axes_grid1 import make_axes_locatable

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


def _to_numpy(d: dict):
    for k, v in d.items():
        if isinstance(v, torch.Tensor):
            d[k] = v.numpy()
        elif isinstance(v, (float, int, np.ndarray)):
            d[k] = v
        elif isinstance(v, dict):
            v = _to_numpy(v)
            d[k] = v
        else:
            msg = f"Data returned can only be dict, tensor, array, or float but got {type(v)}"
            raise TypeError(msg)
    return d


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
        "figure.autolayout": True,
    }
    mpl.rcParams.update(rc_params)
    sns.set_context(rc=rc_params)


class BlissFigure:
    def __init__(
        self,
        figdir: str,
        cachedir: str,
        overwrite: bool = False,
        img_format: str = "png",
    ) -> None:
        self.figdir = Path(figdir)
        self.cachefile = Path(cachedir) / (self.cache_name + ".pt")
        self.overwrite = overwrite
        self.img_format = img_format

    @property
    def rc_kwargs(self) -> dict:
        return {}

    @property
    @abstractmethod
    def cache_name(self) -> str:
        """Unique identifier for set of figures including cache."""
        return "cache_name"

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for set of figures including cache."""
        return "bliss_fig"

    @abstractmethod
    def compute_data(self, *args, **kwargs) -> dict:
        """Should only return tensors that can be casted to numpy."""
        return {}

    @abstractmethod
    def create_figure(self, data) -> Figure:
        """Return matplotlib figure instances to save based on data."""
        return {}

    def get_data(self, *args, **kwargs) -> dict:
        """Return summary of data for producing plot, must be cachable w/ torch.save()."""
        if self.cachefile.exists() and not self.overwrite:
            return torch.load(self.cachefile)

        data = self.compute_data(*args, **kwargs)
        torch.save(data, self.cachefile)
        return data

    def __call__(self, *args, **kwargs):
        """Create figures and save to output directory with names from `self.fignames`."""
        set_rc_params(**self.rc_kwargs)
        data = self.get_data(*args, **kwargs)
        data_np = _to_numpy(data)
        fig: Figure = self.create_figure(data_np)  # data for figures is all numpy arrays or floats.
        figfile = self.figdir / f"{self.name}.{self.img_format}"
        fig.savefig(figfile, format=self.img_format)  # pylint: disable=no-member
        plt.close(fig)


def plot_detections(images, true_cat, est_cat, nrows, img_ids, margin_px):
    fig, axes = plt.subplots(nrows=nrows, ncols=nrows, figsize=(20, 20))
    axes = axes.flatten() if nrows > 1 else [axes]  # flatten

    for ax_idx, ax in enumerate(axes):
        if ax_idx >= len(img_ids):  # don't plot on this ax if there aren't enough images
            break

        img_id = img_ids[ax_idx]
        true_n_sources = true_cat.n_sources[img_id].item()
        n_sources = est_cat.n_sources[img_id].item()
        ax.set_xlabel(f"True num: {true_n_sources}; Est num: {n_sources}")

        # add white border showing where centers of stars and galaxies can be
        ax.axvline(margin_px, color="w")
        ax.axvline(images.shape[-1] - margin_px, color="w")
        ax.axhline(margin_px, color="w")
        ax.axhline(images.shape[-2] - margin_px, color="w")

        # plot image first
        image = images[img_id, 0].cpu().numpy()
        vmin = image.min().item()
        vmax = image.max().item()
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        im = ax.matshow(image, vmin=vmin, vmax=vmax, cmap="viridis")
        fig.colorbar(im, cax=cax, orientation="vertical")

        true_cat.plot_plocs(ax, img_id, "galaxy", bp=margin_px, color="r", marker="x", s=20)
        true_cat.plot_plocs(ax, img_id, "star", bp=margin_px, color="m", marker="x", s=20)
        est_cat.plot_plocs(ax, img_id, "all", bp=margin_px, color="b", marker="+", s=30)

        if ax_idx == 0:
            ax.scatter(None, None, color="r", marker="x", s=20, label="t.gal")
            ax.scatter(None, None, color="m", marker="x", s=20, label="t.star")
            ax.scatter(None, None, color="b", marker="+", s=30, label="p.source")
            ax.legend(
                bbox_to_anchor=(0.0, 1.2, 1.0, 0.102),
                loc="lower left",
                ncol=2,
                mode="expand",
                borderaxespad=0.0,
            )

    fig.tight_layout()
    return fig
