"""Common functions to plot results."""
from abc import abstractmethod
from pathlib import Path
from typing import Optional, Tuple

import matplotlib as mpl
import numpy as np
import seaborn as sns
import torch
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from matplotlib.pyplot import Axes
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


def plot_image(
    fig: Figure,
    ax: Axes,
    image: np.ndarray,
    vrange: Optional[tuple] = None,
    colorbar: bool = True,
    cmap="gray",
) -> None:
    h, w = image.shape
    assert h == w
    vmin = image.min().item() if vrange is None else vrange[0]
    vmax = image.max().item() if vrange is None else vrange[1]

    if colorbar:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
    im = ax.matshow(image, vmin=vmin, vmax=vmax, cmap=cmap)
    if colorbar:
        fig.colorbar(im, cax=cax, orientation="vertical")


def plot_locs(
    ax: Axes,
    bp: int,
    slen: int,
    plocs: np.ndarray,
    galaxy_probs: np.ndarray,
    m: str = "x",
    s: float = 20,
    lw: float = 1,
    alpha: float = 1,
    annotate=False,
    cmap: str = "bwr",
) -> None:
    n_samples, xy = plocs.shape
    assert galaxy_probs.shape == (n_samples,) and xy == 2

    x = plocs[:, 1] - 0.5 + bp
    y = plocs[:, 0] - 0.5 + bp
    for i, (xi, yi) in enumerate(zip(x, y)):
        prob = galaxy_probs[i]
        cmp = mpl.colormaps[cmap]
        color = cmp(prob)
        if bp < xi < slen - bp and bp < yi < slen - bp:
            ax.scatter(xi, yi, color=color, marker=m, s=s, lw=lw, alpha=alpha)
            if annotate:
                ax.annotate(f"{galaxy_probs[i]:.2f}", (xi, yi), color=color, fontsize=8)


def add_loc_legend(ax: mpl.axes.Axes, labels: list, cmap1="cool", cmap2="bwr", s=20):
    cmp1 = mpl.colormaps[cmap1]
    cmp2 = mpl.colormaps[cmap2]
    colors = (cmp1(1.0), cmp1(0.0), cmp2(1.0), cmp2(0.0))
    markers = ("+", "+", "x", "x")
    sizes = (s * 2, s * 2, s + 5, s + 5)
    for label, c, m, size in zip(labels, colors, markers, sizes):
        ax.scatter([], [], color=c, marker=m, label=label, s=size)
    ax.legend(
        bbox_to_anchor=(0.0, 1.2, 1.0, 0.102),
        loc="lower left",
        ncol=2,
        mode="expand",
        borderaxespad=0.0,
    )


def scatter_bin_plot(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    xlims: Tuple[int, int],
    delta: float,
    capsize=5.0,
    qs=(0.25, 0.75),
    color="m",
):
    # plot median and 25/75 quantiles on each bin decided by delta and xlims.
    xbins = np.arange(xlims[0], xlims[1], delta)

    xs = np.zeros(len(xbins))
    ys = np.zeros(len(xbins))
    errs = np.zeros((len(xbins), 2))

    for i, bx in enumerate(xbins):
        keep_x = (x > bx) & (x < bx + delta)
        y_bin: np.ndarray = y[keep_x]

        xs[i] = bx + delta / 2

        if y_bin.shape[0] == 0:
            ys[i] = np.nan
            errs[i] = (np.nan, np.nan)
            continue

        ys[i] = np.median(y_bin)
        errs[i, :] = ys[i] - np.quantile(y_bin, qs[0]), np.quantile(y_bin, qs[1]) - ys[i]

    errs = errs.T.reshape(2, -1)
    ax.errorbar(xs, ys, yerr=errs, marker="o", c=color, linestyle="-", capsize=capsize)


def make_scatter_contours(ax, x, y):
    sns.scatterplot(x=x, y=y, s=10, color="0.15", ax=ax)
    sns.histplot(x=x, y=y, pthresh=0.1, cmap="mako", ax=ax, cbar=True)
    sns.kdeplot(x=x, y=y, levels=10, color="w", linewidths=1, ax=ax)
