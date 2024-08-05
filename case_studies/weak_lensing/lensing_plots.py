import math

import torch
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from torchmetrics import Metric

from bliss.catalog import BaseTileCatalog


class PlotWeakLensingShearConvergence(Metric):
    """Metric wrapper for plotting sample images."""

    def __init__(
        self,
        frequency: int = 1,
        restrict_batch: int = 0,
        tile_slen: int = 0,  # note must match encoder tile_slen
        save_local: str = None,
    ):
        super().__init__()

        self.frequency = frequency
        self.restrict_batch = restrict_batch
        self.tile_slen = tile_slen

        self.should_plot = False
        self.batch = {}
        self.batch_idx = -1
        self.sample_with_mode_tile = None
        self.images = None
        self.current_epoch = 0
        self.target_cat_cropped = None
        self.save_local = save_local

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
        self.sample_with_mode_tile = sample_with_mode_tile
        self.images = batch["images"]
        self.target_cat_cropped = target_cat_cropped

    def compute(self):
        return {}

    def plot(self):
        if self.current_epoch % self.frequency != 0:
            return None
        est_cat = self.sample_with_mode_tile
        true_tile_cat = BaseTileCatalog(self.batch["tile_catalog"])
        return plot_maps(
            self.images,
            true_tile_cat,
            est_cat,
            figsize=None,
            current_epoch=self.current_epoch,
            save_local=self.save_local,
        )


def plot_maps(images, true_tile_cat, est_tile_cat, figsize=None, current_epoch=0, save_local=None):
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
        shear1_vmin = torch.min(true_shear[img_id].squeeze()[:, :, 0])
        shear1_vmax = torch.max(true_shear[img_id].squeeze()[:, :, 0])
        shear2_vmin = torch.min(true_shear[img_id].squeeze()[:, :, 1])
        shear2_vmax = torch.max(true_shear[img_id].squeeze()[:, :, 1])

        convergence_vmin = torch.min(true_convergence[img_id].squeeze())
        convergence_vmax = torch.max(true_convergence[img_id].squeeze())

        plot_maps_helper(
            x_label="True horizontal shear",
            mp=true_shear[img_id].squeeze()[:, :, 0],
            ax=axes[0],
            fig=fig,
            vmin=shear1_vmin,
            vmax=shear1_vmax,
        )
        plot_maps_helper(
            x_label="Estimated horizontal shear",
            mp=est_shear[img_id].squeeze()[:, :, 0],
            ax=axes[1],
            fig=fig,
            vmin=shear1_vmin,
            vmax=shear1_vmax,
        )
        plot_maps_helper(
            x_label="True diagonal shear",
            mp=true_shear[img_id].squeeze()[:, :, 1],
            ax=axes[2],
            fig=fig,
            vmin=shear2_vmin,
            vmax=shear2_vmax,
        )
        plot_maps_helper(
            x_label="Estimated diagonal shear",
            mp=est_shear[img_id].squeeze()[:, :, 1],
            ax=axes[3],
            fig=fig,
            vmin=shear2_vmin,
            vmax=shear2_vmax,
        )
        plot_maps_helper(
            x_label="True convergence",
            mp=true_convergence[img_id].squeeze(),
            ax=axes[4],
            fig=fig,
            vmin=convergence_vmin,
            vmax=convergence_vmax,
        )
        plot_maps_helper(
            x_label="Estimated convergence",
            mp=est_convergence[img_id].squeeze(),
            ax=axes[5],
            fig=fig,
            vmin=convergence_vmin,
            vmax=convergence_vmax,
        )

    fig.tight_layout()
    if save_local:
        fig.savefig(f"{save_local}/wl_shear_conv_{current_epoch}_asinh.png")
    return fig, axes


def plot_maps_helper(x_label: str, mp, ax, fig, vmin, vmax):
    ax.set_xlabel(x_label)

    mp = mp.cpu().numpy()
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    im = ax.matshow(
        mp,
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
    )
    fig.colorbar(im, cax=cax, orientation="vertical")
