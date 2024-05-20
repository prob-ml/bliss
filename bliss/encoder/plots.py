import math
import warnings

import hydra
import numpy as np
import torch
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from omegaconf import DictConfig, ListConfig

from bliss.catalog import TileCatalog


class PlotCollection:
    def __init__(self, plotfns, freqs):
        self.plotfns = plotfns
        self.freqs = freqs

    def plot_one(self, fn_key_or_idx, state_dict, logger=None, **kwargs):
        tag, only_plot = hydra.utils.get_method(self.plotfns[fn_key_or_idx])(
            state_dict=state_dict,
            **kwargs,
        )
        if logger and only_plot:
            logger.experiment.add_figure(tag, only_plot, close=True)
        return tag

    # TODO: address non-descriptive error with mismatched types in instantiation (throws below)
    def plot_all(self, state_dict, check_freqs, logger=None, **kwargs):
        if check_freqs:
            assert (
                "current_iteration" in state_dict
            ), "Plotting frequency enabled but no current iteration counter provided"
            curr_iteration = state_dict["current_iteration"]
            if "max_iterations" not in state_dict:
                max_iterations = -1
                warnings.warn(
                    "Warning: Current iteration set in plotting but max iterations not set."
                    + "Ignore if expected behavior.",
                    RuntimeWarning,
                )
            else:
                max_iterations = state_dict["max_iterations"]
        if isinstance(self.plotfns, ListConfig):
            for idx, freq in enumerate(self.freqs):
                should_plot = (
                    (not check_freqs)
                    or curr_iteration % freq == 0
                    or curr_iteration == max_iterations - 1
                )
                if should_plot:
                    self.plot_one(
                        fn_key_or_idx=idx,
                        state_dict=state_dict,
                        logger=logger,
                        **kwargs,
                    )  # TODO: warn if tag already exists (silent overwriting of plots)
        elif isinstance(self.plotfns, DictConfig):
            for plot_key in self.plotfns.keys():
                should_plot = (
                    not check_freqs
                    or curr_iteration % self.freqs[plot_key] == 0
                    or curr_iteration == max_iterations - 1
                )
                if should_plot:
                    self.plot_one(
                        fn_key_or_idx=plot_key,
                        state_dict=state_dict,
                        logger=logger,
                        **kwargs,
                    )  # TODO: warn if tag already exists (silent overwriting of plots)
        else:
            raise TypeError(
                "Invalid type found for plotting functions collection, "
                + "expected list or dict but found "
                + str(type(self.plotfns))
            )


def plot_sample_images(state_dict, **kwargs):
    batch_restriction = (
        "restrict_batch" in state_dict
        and "batch_idx" in state_dict
        and state_dict["batch_idx"] != state_dict["restrict_batch"]
    )
    if batch_restriction:
        return "", None
    tile_slen = state_dict["tile_slen"]
    batch = state_dict["batch"]
    min_flux_threshold = state_dict["min_flux_threshold"]
    tiles_to_crop = state_dict["tiles_to_crop"]
    sample = state_dict["sample"]
    current_epoch = state_dict["current_iteration"]
    logging_name = state_dict["logging_name"]

    target_cat = TileCatalog(tile_slen, batch["tile_catalog"])
    target_cat = target_cat.filter_tile_catalog_by_flux(min_flux=min_flux_threshold)
    target_cat_cropped = target_cat.symmetric_crop(tiles_to_crop)
    est_cat = sample(batch, use_mode=True)
    mp = tiles_to_crop * tile_slen
    fig = plot_detections(batch["images"], target_cat_cropped, est_cat, margin_px=mp)
    title = f"Epoch:{current_epoch}/{logging_name} images"
    return title, fig


def plot_detections(images, true_tile_cat, est_tile_cat, margin_px, ticks=None, figsize=None):
    """Plots an image of true and estimated sources."""
    batch_size = images.size(0)
    n_samples = min(int(math.sqrt(batch_size)) ** 2, 16)
    nrows = int(n_samples**0.5)
    img_ids = torch.arange(n_samples, device=images.device)

    if figsize is None:
        figsize = (20, 20)
    fig, axes = plt.subplots(nrows=nrows, ncols=nrows, figsize=figsize)
    axes = axes.flatten() if nrows > 1 else [axes]  # flatten

    true_cat = true_tile_cat.to_full_catalog()
    est_cat = est_tile_cat.to_full_catalog()

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
