"""Common functions to plot results."""
import math

import numpy as np
import torch
from matplotlib import pyplot as plt
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


def plot_maps(images, true_tile_cat, est_tile_cat, figsize=None):
    """Plots shear and convergence maps."""
    batch_size = images.size(0)

    n_samples = min(int(math.sqrt(batch_size)) ** 2, 4)

    # every row should be a true map vs generated map for 3 types
    # of maps (shear 1, shear 2, convergence) and the image
    nrows = n_samples
    ncols = 3 * 2 + 1
    img_ids = torch.arange(n_samples, device=images.device)

    if figsize is None:
        # using each image as 5x5 in size:
        figsize = (n_samples * 5, ncols * 5)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    axes = axes.flatten() if nrows > 1 else [axes]  # flatten

    true_shear = true_tile_cat["shear"]
    est_shear = est_tile_cat["shear"]
    true_convergence = true_tile_cat["convergence"]
    est_convergence = est_tile_cat["convergence"]

    for ax_idx, ax in enumerate(axes):
        if ax_idx % 7 == 0:
            # synthetic image
            ax.set_xlabel("Image")
            img_id = img_ids[ax_idx // 7]
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
        if ax_idx % 7 == 1:
            plot_maps_helper(
                x_label="True horizontal shear",
                mp=true_shear[img_id].squeeze()[:, :, 0],
                ax=ax,
                fig=fig,
            )
        if ax_idx % 7 == 2:
            plot_maps_helper(
                x_label="Estimated horizontal shear",
                mp=est_shear[img_id].squeeze()[:, :, 0],
                ax=ax,
                fig=fig,
            )
        if ax_idx % 7 == 3:
            plot_maps_helper(
                x_label="True diagonal shear",
                mp=true_shear[img_id].squeeze()[:, :, 1],
                ax=ax,
                fig=fig,
            )
        if ax_idx % 7 == 4:
            plot_maps_helper(
                x_label="Estimated diagonal shear",
                mp=est_shear[img_id].squeeze()[:, :, 1],
                ax=ax,
                fig=fig,
            )
        if ax_idx % 7 == 5:
            plot_maps_helper(
                x_label="True convergence",
                mp=true_convergence[img_id].squeeze(),
                ax=ax,
                fig=fig,
            )
        if ax_idx % 7 == 6:
            plot_maps_helper(
                x_label="Estimated convergence",
                mp=est_convergence[img_id].squeeze(),
                ax=ax,
                fig=fig,
            )

    fig.tight_layout()
    return fig


def plot_maps_helper(x_label: str, mp, ax, fig):
    ax.set_xlabel(x_label)

    mp = mp.cpu().numpy()
    vmin = mp.min().item()
    vmax = mp.max().item()
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    im = ax.matshow(
        mp,
        vmin=vmin,
        vmax=vmax,
        cmap="viridis",
        extent=(0, mp.shape[0], mp.shape[1], 0),
    )
    fig.colorbar(im, cax=cax, orientation="vertical")
