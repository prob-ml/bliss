"""Common functions to plot results."""
from typing import Dict, Tuple

import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from mpl_toolkits.axes_grid1 import make_axes_locatable


def plot_plocs(catalog, ax, idx, filter_by="all", bp=0, **kwargs):
    """Plots pixel coordinates of sources on given axis.

    Args:
        catalog: FullCatalog to get sources from
        ax: Matplotlib axes to plot on
        idx: index of the image in the batch to plot sources of
        filter_by: Which sources to plot. Can be either a string specifying the source type (either
            'galaxy', 'star', or 'all'), or a list of indices. Defaults to "all".
        bp: Offset added to locations, used for adjusting for cropped tiles. Defaults to 0.
        **kwargs: Keyword arguments to pass to scatter plot.

    Raises:
        NotImplementedError: If object_type is not one of 'galaxy', 'star', 'all', or a List.
    """
    match filter_by:
        case "galaxy":
            keep = catalog.galaxy_bools[idx, :].squeeze(-1).bool()
        case "star":
            keep = catalog.star_bools[idx, :].squeeze(-1).bool()
        case "all":
            keep = torch.ones(catalog.max_sources, dtype=torch.bool, device=catalog.plocs.device)
        case list():
            keep = filter_by
        case _:
            raise NotImplementedError(f"Unknown filter option {filter} specified")

    plocs = catalog.plocs[idx, keep] + bp
    plocs = plocs.detach().cpu()
    ax.scatter(plocs[:, 1], plocs[:, 0], **kwargs)


def plot_detections(images, true_cat, est_cat, nrows, img_ids, margin_px, ticks=None, figsize=None):
    """Plots an image of true and estimated sources."""
    if figsize is None:
        figsize = (20, 20)
    fig, axes = plt.subplots(nrows=nrows, ncols=nrows, figsize=figsize)
    axes = axes.flatten() if nrows > 1 else [axes]  # flatten

    for ax_idx, ax in enumerate(axes):
        if ax_idx >= len(img_ids):  # don't plot on this ax if there aren't enough images
            break

        img_id = img_ids[ax_idx]
        true_n_sources = int(true_cat.n_sources[img_id].item())
        n_sources = int(est_cat.n_sources[img_id].item())
        ax.set_xlabel(f"True num: {true_n_sources}; Est num: {n_sources}")

        # add white border showing where centers of stars and galaxies can be
        if margin_px > 0:
            ax.axvline(margin_px, color="w")
            ax.axvline(images.shape[-1] - margin_px, color="w")
            ax.axhline(margin_px, color="w")
            ax.axhline(images.shape[-2] - margin_px, color="w")

        # plot image first
        image = images[img_id].cpu().numpy()
        image = np.sum(image, axis=0)
        vmin = image.min().item()
        vmax = image.max().item()
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        im = ax.matshow(
            image,
            vmin=vmin,
            vmax=vmax,
            cmap="viridis",
            extent=(0, image.shape[0], image.shape[1], 0),
        )
        fig.colorbar(im, cax=cax, orientation="vertical")

        plot_plocs(true_cat, ax, img_id, "galaxy", bp=margin_px, color="r", marker="x", s=20)
        plot_plocs(true_cat, ax, img_id, "star", bp=margin_px, color="m", marker="x", s=20)
        plot_plocs(est_cat, ax, img_id, "all", bp=margin_px, color="b", marker="+", s=30)

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

        if ticks is not None:
            xticks = list(set(ticks[:, 1].tolist()))
            yticks = list(set(ticks[:, 0].tolist()))

            ax.set_xticks(xticks, [f"{val:.1f}" for val in xticks], fontsize=8)
            ax.set_yticks(yticks, [f"{val:.1f}" for val in yticks], fontsize=8)
            ax.grid(linestyle="dotted", which="major")

    fig.tight_layout()
    return fig


def create_detection_figure(data) -> Figure:
    # take middle of bin as x for plotting
    snr_bins = np.log10(data["detection"]["bins"].mean(1))
    return _make_pr_figure(
        snr_bins, data["detection"], r"$\log_{10} \rm SNR$", xlims=(0.5, 3), ylims2=(0, 2000)
    )


def compute_pr(tgbool: np.ndarray, egbool: np.ndarray):
    t = np.sum(tgbool)
    p = np.sum(egbool)

    cond1 = np.equal(tgbool, egbool).astype(bool)
    cond2 = tgbool.astype(bool)
    tp = (cond1 & cond2).astype(float).sum()

    assert np.all(np.greater_equal(t, tp))
    assert np.all(np.greater_equal(p, tp))
    if t == 0 or p == 0:
        return np.nan, np.nan

    return tp / p, tp / t


def create_classification_figure(data) -> Figure:
    snrs, _, tgbools, egbools = data["classification"].values()
    snr_bins = data["detection"]["bins"]
    n_matches = len(snrs)
    n_bins = len(snr_bins)
    n_boots = 1000

    precision = np.zeros(n_bins)
    recall = np.zeros(n_bins)
    tgals = np.zeros(n_bins)
    egals = np.zeros(n_bins)
    tstars = np.zeros(n_bins)
    estars = np.zeros(n_bins)

    boot_precision = np.zeros((n_boots, n_bins))
    boot_recall = np.zeros((n_boots, n_bins))

    boot_indices = np.random.randint(0, n_matches, (n_boots, n_matches))

    # compute boostrap precision and recall per bin
    for ii in range(n_boots):
        snrs_ii = snrs[boot_indices[ii]]
        tgbools_ii = tgbools[boot_indices[ii]]
        egbools_ii = egbools[boot_indices[ii]]
        for jj, (b1, b2) in enumerate(snr_bins):
            keep = (b1 < snrs_ii) & (snrs_ii < b2)
            tgbool_ii = tgbools_ii[keep]
            egbool_ii = egbools_ii[keep]

            p, r = compute_pr(tgbool_ii, egbool_ii)
            boot_precision[ii][jj] = p
            boot_recall[ii][jj] = r

    # compute precision and recall per bin
    for jj, (b1, b2) in enumerate(snr_bins):
        keep = (b1 < snrs) & (snrs < b2)
        tgbool = tgbools[keep]
        egbool = egbools[keep]
        p, r = compute_pr(tgbool, egbool)
        precision[jj] = p
        recall[jj] = r

        tgals[jj] = tgbool.sum()
        egals[jj] = egbool.sum()
        tstars[jj] = (~tgbool.astype(bool)).astype(float).sum()
        estars[jj] = (~egbool.astype(bool)).astype(float).sum()

    bins = np.log10(snr_bins.mean(1))
    data = {
        "precision": precision,
        "recall": recall,
        "tgcount": tgals,
        "egcount": egals,
        "tscount": tstars,
        "escount": estars,
        "boot": {"precision": boot_precision, "recall": boot_recall},
    }
    return _make_pr_figure(
        bins,
        data,
        r"$\log_{10} \rm SNR$",
        xlims=(0.5, 3),
        metric_type="Galaxy Classification",
        ylims2=(0, 1000),
        legend_size_hist=16,
        ylims=(0.5, 1.03),
    )


def _make_pr_figure(
    bins: np.ndarray,
    data: Dict[str, np.ndarray],
    xlabel: str,
    xlims: Tuple[float, float] = None,
    ylims: Tuple[float, float] = None,
    ylims2: Tuple[float, float] = None,
    ratio: float = 2,
    where_step: str = "mid",
    n_ticks: int = 5,
    ordmag: int = 3,
    metric_type: str = "Detection",
    legend_size_hist: int = 20,
):
    precision = data["precision"]
    recall = data["recall"]
    boot_precision = data["boot"]["precision"]
    boot_recall = data["boot"]["recall"]
    tgcount = data["tgcount"]
    tscount = data["tscount"]
    egcount = data["egcount"]
    escount = data["escount"]

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(10, 10), gridspec_kw={"height_ratios": [1, ratio]}, sharex=True
    )

    # (bottom) plot of precision and recall
    ymin = min(precision.min(), recall.min())
    yticks = np.arange(np.round(ymin, 1), 1.1, 0.1)
    c1 = plt.rcParams["axes.prop_cycle"].by_key()["color"][0]
    precision1 = np.quantile(boot_precision, 0.25, 0)
    precision2 = np.quantile(boot_precision, 0.75, 0)
    ax2.plot(bins, precision, "-o", color=c1, label=r"\rm Precision", markersize=6)
    ax2.fill_between(bins, precision1, precision2, color=c1, alpha=0.5)

    c2 = plt.rcParams["axes.prop_cycle"].by_key()["color"][1]
    recall1 = np.quantile(boot_recall, 0.25, 0)
    recall2 = np.quantile(boot_recall, 0.75, 0)
    ax2.plot(bins, recall, "-o", color=c2, label=r"\rm Recall", markersize=6)
    ax2.fill_between(bins, recall1, recall2, color=c2, alpha=0.5)

    ax2.legend()
    ax2.set_xlabel(xlabel)
    ax2.set_ylabel(rf"\rm {metric_type} Metric")
    ax2.set_yticks(yticks)
    ax2.grid(linestyle="-", linewidth=0.5, which="major", axis="both")

    if xlims is not None:
        ax2.set_xlim(xlims)
    if ylims is not None:
        ax2.set_ylim(ylims)
    if ylims2 is not None:
        ax1.set_ylim(ylims2)

    # setup histogram plot up top
    c1 = plt.rcParams["axes.prop_cycle"].by_key()["color"][3]
    c2 = plt.rcParams["axes.prop_cycle"].by_key()["color"][4]
    ax1.step(bins, tgcount, label="True galaxies", where=where_step, color=c1)
    ax1.step(bins, tscount, label="True stars", where=where_step, color=c2)
    ax1.step(bins, egcount, label="Pred. galaxies", ls="--", where=where_step, color=c1)
    ax1.step(bins, escount, label="Pred. stars", ls="--", where=where_step, color=c2)
    ymax = max(tgcount.max(), tscount.max(), egcount.max(), escount.max())
    yticks = np.round(np.linspace(0, ymax, n_ticks), -ordmag)
    ax1.set_yticks(yticks)
    ax1.set_ylabel(r"\rm Counts")
    ax1.legend(loc="best", prop={"size": legend_size_hist})
    ax1.grid(linestyle="-", linewidth=0.5, which="major", axis="both")
    plt.subplots_adjust(hspace=0)
    return fig
