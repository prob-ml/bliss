from pathlib import Path

import torch
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from torchmetrics import Metric


class PlotWeakLensingShearConvergence(Metric):
    """Metric wrapper for plotting sample images."""

    def __init__(
        self,
        frequency: int = 1,
        restrict_batch: int = 0,
        save_local: str = None,
    ):
        super().__init__()

        self.frequency = frequency
        self.restrict_batch = restrict_batch

        self.should_plot = False
        self.batch_idx = -1
        self.true_shear1 = None
        self.true_shear2 = None
        self.true_convergence = None
        self.est_shear1 = None
        self.est_shear2 = None
        self.est_convergence = None
        self.current_epoch = 0
        self.save_local = save_local

    def update(
        self,
        target_cat_cropped,
        sample_with_mode_tile,
        current_epoch,
        batch_idx,
    ):
        self.batch_idx = batch_idx
        if self.restrict_batch != batch_idx:
            self.should_plot = False
            return
        self.current_epoch = current_epoch
        self.should_plot = True
        self.true_shear1 = target_cat_cropped["shear_1"]
        self.true_shear2 = target_cat_cropped["shear_2"]
        self.true_convergence = target_cat_cropped["convergence"]
        self.est_shear1 = sample_with_mode_tile.get("shear_1", torch.zeros_like(self.true_shear1))
        self.est_shear2 = sample_with_mode_tile.get("shear_2", torch.zeros_like(self.true_shear2))
        self.est_convergence = sample_with_mode_tile.get(
            "convergence", torch.zeros_like(self.true_convergence)
        )

    def compute(self):
        return {}

    def plot(self):
        if self.current_epoch % self.frequency != 0:
            return None
        return plot_maps(
            self.true_shear1,
            self.true_shear2,
            self.true_convergence,
            self.est_shear1,
            self.est_shear2,
            self.est_convergence,
            figsize=None,
            current_epoch=self.current_epoch,
            save_local=self.save_local,
        )


def plot_maps(
    true_shear1,
    true_shear2,
    true_convergence,
    est_shear1,
    est_shear2,
    est_convergence,
    figsize=None,
    current_epoch=0,
    save_local=None,
):
    """Plots weak lensing shear and convergence maps."""
    num_images = 1
    num_lensing_params = 6  # true and estimated shear1, shear2, and convergence
    img_ids = torch.arange(num_images)

    if figsize is None:
        figsize = (20, 20)
    fig, axes = plt.subplots(nrows=num_images, ncols=num_lensing_params, figsize=figsize)

    for img_id in img_ids:
        plot_maps_helper(
            label="True shear 1",
            mp=true_shear1[img_id].squeeze(),
            ax=axes[0],
            fig=fig,
        )
        plot_maps_helper(
            label="Estimated shear 1",
            mp=est_shear1[img_id].squeeze(),
            ax=axes[1],
            fig=fig,
        )
        plot_maps_helper(
            label="True shear 2",
            mp=true_shear2[img_id].squeeze(),
            ax=axes[2],
            fig=fig,
        )
        plot_maps_helper(
            label="Estimated shear 2",
            mp=est_shear2[img_id].squeeze(),
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
