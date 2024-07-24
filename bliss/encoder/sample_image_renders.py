import math

import numpy as np
import torch
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from torchmetrics import Metric

from bliss.catalog import FullCatalog


class PlotSampleImages(Metric):
    """Metric wrapper for plotting sample images."""

    def __init__(
        self,
        frequency: int = 1,
        restrict_batch: int = 0,
        tiles_to_crop: int = 0,
        tile_slen: int = 0,  # note must match encoder tile_slen
    ):
        super().__init__()

        self.frequency = frequency
        self.restrict_batch = restrict_batch
        self.tiles_to_crop = tiles_to_crop
        self.tile_slen = tile_slen

        self.should_plot = False
        self.batch = {}
        self.batch_idx = -1
        self.sample_with_mode = None
        self.images = None
        self.current_epoch = 0
        self.target_cat_cropped = None

    def update(
        self,
        batch,
        target_cat_cropped,
        sample_with_mode_tile,
        sample_with_mode,
        current_epoch,
        batch_idx,
    ):
        self.batch_idx = batch_idx
        if self.restrict_batch != batch_idx:
            self.should_plot = False
            return
        self.current_epoch = current_epoch
        self.should_plot = True
        self.batch = batch
        self.sample_with_mode = sample_with_mode
        self.images = batch["images"]
        self.target_cat_cropped = target_cat_cropped

    def compute(self):
        return {}

    def plot(self):
        mp = self.tile_slen * self.tiles_to_crop
        if self.current_epoch % self.frequency != 0:
            return None
        est_cat = self.sample_with_mode
        return plot_detections(self.images, self.target_cat_cropped, est_cat, margin_px=mp)


def plot_detections(images, true_cat, est_cat, margin_px, ticks=None, figsize=None):
    """Plots an image of true and estimated sources."""
    assert isinstance(true_cat, FullCatalog)
    assert isinstance(est_cat, FullCatalog)

    batch_size = images.size(0)
    n_samples = min(int(math.sqrt(batch_size)) ** 2, 16)
    nrows = int(n_samples**0.5)
    img_ids = torch.arange(n_samples, device=images.device)

    if figsize is None:
        figsize = (20, 20)
    fig, axes = plt.subplots(nrows=nrows, ncols=nrows, figsize=figsize)
    axes = axes.flatten() if nrows > 1 else [axes]  # flatten

    for ax_idx, ax in enumerate(axes):
        if ax_idx >= len(img_ids):  # don't plot on this ax if there aren't enough images
            break

        img_id = img_ids[ax_idx]
        true_n_sources = int(true_cat["n_sources"][img_id].item())
        n_sources = int(est_cat["n_sources"][img_id].item())
        ax.set_xlabel(f"True num: {true_n_sources}; Est num: {n_sources}")

        # add white border showing where centers of stars and galaxies can be
        if margin_px > 0:
            ax.axvline(margin_px, color="w")
            ax.axvline(images.shape[-1] - margin_px, color="w")
            ax.axhline(margin_px, color="w")
            ax.axhline(images.shape[-2] - margin_px, color="w")

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
    return fig, axes


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
            keep = torch.ones(catalog.max_sources, dtype=torch.bool, device=catalog["plocs"].device)
        case list():
            keep = filter_by
        case _:
            raise NotImplementedError(f"Unknown filter option {filter} specified")

    plocs = catalog["plocs"][idx, keep] + bp
    plocs = plocs.detach().cpu()
    ax.scatter(plocs[:, 1], plocs[:, 0], **kwargs)
