import math
from pathlib import Path

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

    true_shear1 = true_tile_cat["shear_1"]
    true_shear2 = true_tile_cat["shear_2"]
    pred_shear1 = est_tile_cat["shear_1"]
    pred_shear2 = est_tile_cat["shear_2"]
    true_shear = torch.cat((true_shear1, true_shear2), dim=-1)
    est_shear = torch.cat((pred_shear1, pred_shear2), dim=-1)

    if "convergence" not in est_tile_cat:
        true_convergence = torch.zeros_like(true_shear1)
        est_convergence = torch.zeros_like(true_convergence)
    else:
        true_convergence = true_tile_cat["convergence"]
        est_convergence = est_tile_cat["convergence"]

    for img_id in img_ids:
        plot_maps_helper(
            label="True shear 1",
            mp=true_shear[img_id].squeeze()[:, :, 0],
            ax=axes[0],
            fig=fig,
        )
        plot_maps_helper(
            label="Estimated shear 1",
            mp=est_shear[img_id].squeeze()[:, :, 0],
            ax=axes[1],
            fig=fig,
        )
        plot_maps_helper(
            label="True shear 2",
            mp=true_shear[img_id].squeeze()[:, :, 1],
            ax=axes[2],
            fig=fig,
        )
        plot_maps_helper(
            label="Estimated shear 2",
            mp=est_shear[img_id].squeeze()[:, :, 1],
            ax=axes[3],
            fig=fig,
        )
        plot_maps_helper(
            label="True convergence",
            mp=true_convergence[img_id].squeeze(),
            ax=axes[4],
            fig=fig,
        )
        plot_maps_helper(
            label="Estimated convergence",
            mp=est_convergence[img_id].squeeze(),
            ax=axes[5],
            fig=fig,
        )

    fig.tight_layout()
    if save_local:
        if not Path(save_local).exists():
            Path(save_local).mkdir(parents=True)
        fig.savefig(f"{save_local}/lensing_maps_{current_epoch}.png")
    return fig, axes


def plot_maps_helper(label: str, mp, ax, fig):
    ax.set_title(label)

    mp = mp.cpu().float().numpy()
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    im = ax.matshow(
        mp,
        cmap="viridis",
    )
    fig.colorbar(im, cax=cax, orientation="vertical")
