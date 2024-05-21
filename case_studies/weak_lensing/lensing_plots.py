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
    """Plots weak lensing shear and convergence maps."""
    batch_size = images.size(0)

    num_images = min(int(math.sqrt(batch_size)) ** 2, 5)
    num_lensing_params = 6  # true and estimated shear1, shear2, and convergence
    img_ids = torch.arange(num_images, device=images.device)

    if figsize is None:
        figsize = (20, 20)
    fig, axes = plt.subplots(nrows=num_images, ncols=num_lensing_params, figsize=figsize)

    true_shear = true_tile_cat["shear"]
    est_shear = est_tile_cat["shear"]
    true_convergence = true_tile_cat["convergence"]
    est_convergence = est_tile_cat["convergence"]

    for img_id in img_ids:
        plot_maps_helper(
            x_label="True horizontal shear",
            mp=true_shear[img_id].squeeze()[:, :, 0],
            ax=axes[img_id, 0],
            fig=fig,
        )
        plot_maps_helper(
            x_label="Estimated horizontal shear",
            mp=est_shear[img_id].squeeze()[:, :, 0],
            ax=axes[img_id, 1],
            fig=fig,
        )
        plot_maps_helper(
            x_label="True diagonal shear",
            mp=true_shear[img_id].squeeze()[:, :, 1],
            ax=axes[img_id, 2],
            fig=fig,
        )
        plot_maps_helper(
            x_label="Estimated diagonal shear",
            mp=est_shear[img_id].squeeze()[:, :, 1],
            ax=axes[img_id, 3],
            fig=fig,
        )
        plot_maps_helper(
            x_label="True convergence",
            mp=true_convergence[img_id].squeeze(),
            ax=axes[img_id, 4],
            fig=fig,
        )
        plot_maps_helper(
            x_label="Estimated convergence",
            mp=est_convergence[img_id].squeeze(),
            ax=axes[img_id, 5],
            fig=fig,
        )

    fig.tight_layout()
    return fig


def plot_maps_helper(x_label: str, mp, ax, fig):
    ax.set_xlabel(x_label)

    mp = mp.cpu().numpy()
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    im = ax.matshow(
        mp,
        cmap="viridis",
        extent=(0, mp.shape[0], mp.shape[1], 0),
    )
    fig.colorbar(im, cax=cax, orientation="vertical")
