from pathlib import Path

import torch
from matplotlib import pyplot as plt
from torchmetrics import Metric


class PlotLensingMaps(Metric):
    """Metric wrapper for plotting sample images."""

    def __init__(
        self,
        frequency: int = 1,
        save_local: str = None,
    ):
        super().__init__()

        self.frequency = frequency

        self.should_plot = False
        self.batch_idx = -1
        self.true_shear1 = []
        self.true_shear2 = []
        self.true_convergence = []
        self.est_shear1 = []
        self.est_shear2 = []
        self.est_convergence = []
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
        self.current_epoch = current_epoch
        self.should_plot = True
        self.true_shear1.append(target_cat_cropped["shear_1"])
        self.true_shear2.append(target_cat_cropped["shear_2"])
        self.true_convergence.append(target_cat_cropped["convergence"])
        self.est_shear1.append(
            sample_with_mode_tile.get("shear_1", torch.zeros_like(target_cat_cropped["shear_1"]))
        )
        self.est_shear2.append(
            sample_with_mode_tile.get("shear_2", torch.zeros_like(target_cat_cropped["shear_2"]))
        )
        self.est_convergence.append(
            sample_with_mode_tile.get(
                "convergence", torch.zeros_like(target_cat_cropped["convergence"])
            )
        )

    def compute(self):
        return {}

    def plot(self):
        if self.current_epoch % self.frequency != 0:
            return None
        true_shear1 = torch.stack(self.true_shear1, dim=0).squeeze()
        true_shear2 = torch.stack(self.true_shear2, dim=0).squeeze()
        true_convergence = torch.stack(self.true_convergence, dim=0).squeeze()
        est_shear1 = torch.stack(self.est_shear1, dim=0).squeeze()
        est_shear2 = torch.stack(self.est_shear2, dim=0).squeeze()
        est_convergence = torch.stack(self.est_convergence, dim=0).squeeze()

        self.true_shear1 = []
        self.true_shear2 = []
        self.true_convergence = []
        self.est_shear1 = []
        self.est_shear2 = []
        self.est_convergence = []

        return plot_maps(
            true_shear1,
            true_shear2,
            true_convergence,
            est_shear1,
            est_shear2,
            est_convergence,
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
    current_epoch=0,
    save_local=None,
):
    """Plots weak lensing shear and convergence maps."""
    num_images = min(4, true_shear1.shape[0])
    num_lensing_params = 6  # true and estimated shear1, shear2, and convergence
    img_ids = torch.arange(num_images)

    fig, axes = plt.subplots(
        nrows=num_images, ncols=num_lensing_params, figsize=(20, 3 * num_images)
    )

    for img_id in img_ids:
        ts1 = axes[img_id, 0].imshow(true_shear1[img_id].squeeze().cpu())
        axes[img_id, 0].set_title("True shear 1")
        axes[img_id, 0].set_ylabel(f"Image {img_id}")
        plt.colorbar(ts1, fraction=0.045)

        es1 = axes[img_id, 1].imshow(est_shear1[img_id].squeeze().cpu())
        axes[img_id, 1].set_title("Estimated shear 1")
        plt.colorbar(es1, fraction=0.045)

        ts2 = axes[img_id, 2].imshow(true_shear2[img_id].squeeze().cpu())
        axes[img_id, 2].set_title("True shear 2")
        plt.colorbar(ts2, fraction=0.045)

        es2 = axes[img_id, 3].imshow(est_shear2[img_id].squeeze().cpu())
        axes[img_id, 3].set_title("Estimated shear 2")
        plt.colorbar(es2, fraction=0.045)

        tc = axes[img_id, 4].imshow(true_convergence[img_id].squeeze().cpu())
        axes[img_id, 4].set_title("True convergence")
        plt.colorbar(tc, fraction=0.045)

        ec = axes[img_id, 5].imshow(est_convergence[img_id].squeeze().cpu())
        axes[img_id, 5].set_title("Estimated convergence")
        plt.colorbar(ec, fraction=0.045)

    fig.tight_layout()

    if save_local:
        if not Path(save_local).exists():
            Path(save_local).mkdir(parents=True)
        fig.savefig(f"{save_local}/lensing_maps_{current_epoch}.png")
    return fig, axes
