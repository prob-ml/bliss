import math

import torch
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from bliss.catalog import TileCatalog


def plot_lensing_maps(state_dict, **kwargs):
    if "restrict_batch" in state_dict and "batch_idx" in state_dict:
        if state_dict["batch_idx"] != state_dict["restrict_batch"]:
            return "", None
    logging_name = state_dict["logging_name"]
    current_epoch = state_dict["current_iteration"]
    sample = state_dict["sample"]
    tiles_to_crop = state_dict["tiles_to_crop"]
    min_flux_threshold = state_dict["min_flux_threshold"]
    batch = state_dict["batch"]
    tile_slen = state_dict["tile_slen"]

    target_cat = TileCatalog(tile_slen, batch["tile_catalog"])
    target_cat = target_cat.filter_tile_catalog_by_flux(min_flux=min_flux_threshold)
    target_cat_cropped = target_cat.symmetric_crop(tiles_to_crop)
    est_cat = sample(batch, use_mode=True)
    fig = plot_maps(batch["images"], target_cat_cropped, est_cat, figsize=None)
    title = f"Epoch:{current_epoch}/{logging_name} shear and convergence"
    return title, fig


def plot_maps(images, true_tile_cat, est_tile_cat, figsize=None):
    """Plots shear and convergence maps."""
    batch_size = images.size(0)

    n_samples = min(int(math.sqrt(batch_size)) ** 2, 4)

    # every row should be a true map vs generated map for 3 types
    # of maps (shear 1, shear 2, convergence) and the image
    nrows = n_samples
    ncols = 3 * 2
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
        img_id = img_ids[ax_idx // 6]
        if ax_idx % 6 == 0:
            plot_maps_helper(
                x_label="True horizontal shear",
                mp=true_shear[img_id].squeeze()[:, :, 0],
                ax=ax,
                fig=fig,
            )
        if ax_idx % 6 == 1:
            plot_maps_helper(
                x_label="Estimated horizontal shear",
                mp=est_shear[img_id].squeeze()[:, :, 0],
                ax=ax,
                fig=fig,
            )
        if ax_idx % 6 == 2:
            plot_maps_helper(
                x_label="True diagonal shear",
                mp=true_shear[img_id].squeeze()[:, :, 1],
                ax=ax,
                fig=fig,
            )
        if ax_idx % 6 == 3:
            plot_maps_helper(
                x_label="Estimated diagonal shear",
                mp=est_shear[img_id].squeeze()[:, :, 1],
                ax=ax,
                fig=fig,
            )
        if ax_idx % 6 == 4:
            plot_maps_helper(
                x_label="True convergence",
                mp=true_convergence[img_id].squeeze(),
                ax=ax,
                fig=fig,
            )
        if ax_idx % 6 == 5:
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
