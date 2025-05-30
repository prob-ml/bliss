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
        self.est_shear1.append(
            sample_with_mode_tile.get("shear_1", torch.zeros_like(target_cat_cropped["shear_1"]))
        )

        self.true_shear2.append(target_cat_cropped["shear_2"])
        self.est_shear2.append(
            sample_with_mode_tile.get("shear_2", torch.zeros_like(target_cat_cropped["shear_2"]))
        )

        if "convergence" not in target_cat_cropped:
            self.true_convergence.append(torch.zeros_like(target_cat_cropped["shear_1"]))
            self.est_convergence.append(torch.zeros_like(target_cat_cropped["shear_1"]))
        else:
            self.true_convergence.append(target_cat_cropped["convergence"])
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

        plot_lensing_scatterplots(
            true_shear1,
            true_shear2,
            true_convergence,
            est_shear1,
            est_shear2,
            est_convergence,
            current_epoch=self.current_epoch,
            save_local=self.save_local,
        )

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
    num_lensing_params = 6  # true and estimated shear1, shear2, and convergence
    num_redshift_bins = true_shear1.shape[-1]

    # randomly select two images to plot
    num_images = 2
    img_idx = torch.randint(0, true_shear1.shape[0], size=[num_images])

    fig, ax = plt.subplots(
        nrows=num_images * num_redshift_bins, ncols=num_lensing_params, figsize=(20, 20)
    )

    for i, idx in enumerate(img_idx):
        for b in range(num_redshift_bins):
            for col in range(num_lensing_params):
                ax[i * num_redshift_bins + b, col].set_xticks([])
                ax[i * num_redshift_bins + b, col].set_yticks([])
            ts1 = ax[i * num_redshift_bins + b, 0].imshow(
                true_shear1[idx, ..., b].cpu(), cmap="coolwarm", vmin=-0.03, vmax=0.03
            )
            ax[i * num_redshift_bins + b, 0].set_title(f"True gamma1, redshift bin {b}")
            ax[i * num_redshift_bins + b, 0].set_ylabel(f"Image {idx}")
            plt.colorbar(ts1, fraction=0.045)

            es1 = ax[i * num_redshift_bins + b, 1].imshow(
                est_shear1[idx, ..., b].cpu(), cmap="coolwarm", vmin=-0.03, vmax=0.03
            )
            ax[i * num_redshift_bins + b, 1].set_title(f"Est. gamma1, redshift bin {b}")
            plt.colorbar(es1, fraction=0.045)

            ts2 = ax[i * num_redshift_bins + b, 2].imshow(
                true_shear2[idx, ..., b].cpu(), cmap="coolwarm", vmin=-0.03, vmax=0.03
            )
            ax[i * num_redshift_bins + b, 2].set_title(f"True gamma2, redshift bin {b}")
            plt.colorbar(ts2, fraction=0.045)

            es2 = ax[i * num_redshift_bins + b, 3].imshow(
                est_shear2[idx, ..., b].cpu(), cmap="coolwarm", vmin=-0.03, vmax=0.03
            )
            ax[i * num_redshift_bins + b, 3].set_title(f"Est. gamma2, redshift bin {b}")
            plt.colorbar(es2, fraction=0.045)

            tc = ax[i * num_redshift_bins + b, 4].imshow(
                true_convergence[idx, ..., b].cpu(), cmap="coolwarm", vmin=-0.03, vmax=0.03
            )
            ax[i * num_redshift_bins + b, 4].set_title(f"True kappa, redshift bin {b}")
            plt.colorbar(tc, fraction=0.045)

            ec = ax[i * num_redshift_bins + b, 5].imshow(
                est_convergence[idx, ..., b].cpu(), cmap="coolwarm", vmin=-0.03, vmax=0.03
            )
            ax[i * num_redshift_bins + b, 5].set_title(f"Est. kappa, redshift bin {b}")
            plt.colorbar(ec, fraction=0.045)

    fig.tight_layout()

    if save_local:
        if not Path(save_local).exists():
            Path(save_local).mkdir(parents=True)
        fig.savefig(f"{save_local}/lensing_maps_{current_epoch}.png")

    plt.close(fig)


def plot_lensing_scatterplots(
    true_shear1,
    true_shear2,
    true_convergence,
    est_shear1,
    est_shear2,
    est_convergence,
    current_epoch=0,
    save_local=None,
):
    """Creates scatterplots of true vs. estimated shear1, shear2, and convergence."""
    num_lensing_params = 3  # shear1, shear2, and convergence
    num_redshift_bins = true_shear1.shape[-1]

    fig, axes = plt.subplots(nrows=num_redshift_bins, ncols=num_lensing_params, figsize=(20, 20))

    axes[0].scatter(true_shear1.flatten().cpu(), est_shear1.flatten().cpu(), alpha=0.2)
    axes[0].set_xlabel("True shear 1")
    axes[0].set_ylabel("Estimated shear 1")
    axes[0].axline((0, 0), slope=1, color="black", linestyle="dashed")

    axes[1].scatter(true_shear2.flatten().cpu(), est_shear2.flatten().cpu(), alpha=0.2)
    axes[1].set_xlabel("True shear 2")
    axes[1].set_ylabel("Estimated shear 2")
    axes[1].axline((0, 0), slope=1, color="black", linestyle="dashed")

    axes[2].scatter(true_convergence.flatten().cpu(), est_convergence.flatten().cpu(), alpha=0.2)
    axes[2].set_xlabel("True convergence")
    axes[2].set_ylabel("Estimated convergence")
    axes[2].axline((0, 0), slope=1, color="black", linestyle="dashed")

    fig.tight_layout()

    if save_local:
        if not Path(save_local).exists():
            Path(save_local).mkdir(parents=True)
        fig.savefig(f"{save_local}/lensing_scatterplots_{current_epoch}.png")

    plt.close(fig)
