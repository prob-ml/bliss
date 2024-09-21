import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.figure import Figure
from torch import Tensor
from tqdm import tqdm

from bliss.encoders.autoencoder import OneCenteredGalaxyAE
from bliss.plotting import BlissFigure, plot_image, scatter_shade_plot
from bliss.reporting import get_single_galaxy_measurements, get_snr


class AutoEncoderFigures(BlissFigure):
    """Figures related to trained autoencoder model."""

    def __init__(self, *args, n_examples: int = 5, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.n_examples = n_examples

    @property
    def all_rcs(self):
        rc_recon = {"fontsize": 22, "tick_label_size": "small", "legend_fontsize": "small"}
        return {
            "ae_random_recon": rc_recon,
            "ae_worst_recon": rc_recon,
            "ae_bin_measurements": {"fontsize": 24},
            "ae_bin_hists": {"fontsize": 24},
        }

    @property
    def fignames(self) -> tuple[str, ...]:
        return ("ae_random_recon", "ae_worst_recon", "ae_bin_measurements", "ae_bin_hists")

    @property
    def cache_name(self) -> str:
        return "ae"

    def compute_data(self, autoencoder: OneCenteredGalaxyAE, images_file: str):
        device = autoencoder.device  # GPU is better otherwise slow.
        image_data = torch.load(images_file)
        images: Tensor = image_data["images"].float()  # for NN
        backgrounds = image_data["background"].float()  # for NN
        background: Tensor = backgrounds[0].reshape(1, 1, 53, 53).to(device)
        noiseless_images: Tensor = image_data["noiseless"].float()  # for NN

        snr: Tensor = get_snr(noiseless_images, backgrounds)
        recon_means = torch.tensor([])

        print("INFO: Computing reconstructions from saved autoencoder model...")
        n_images = images.shape[0]
        batch_size = 128
        n_iters = int(np.ceil(n_images // 128)) + 1
        with torch.no_grad():
            for i in tqdm(range(n_iters)):  # in batches otherwise GPU error.
                bimages = images[batch_size * i : batch_size * (i + 1)].to(device)
                recon_mean = autoencoder.forward(bimages, background)
                recon_mean = recon_mean.detach().to("cpu")
                recon_means = torch.cat((recon_means, recon_mean))
        residuals = (recon_means - images) / recon_means.sqrt()
        assert recon_means.shape[0] == noiseless_images.shape[0]

        # random
        snr_thres = 10
        high_snr_indices = torch.where(snr > snr_thres)[0]
        rand_perm_indx = torch.randperm(len(high_snr_indices))[: self.n_examples]
        rand_indices = high_snr_indices[rand_perm_indx]

        # worst
        q = 100
        absolute_residual = residuals.abs().sum(axis=(1, 2, 3))
        worst_indices = absolute_residual.argsort()[-self.n_examples - q : -q]

        recon_no_background = recon_means - background.cpu()
        assert torch.all(recon_no_background.sum(axis=(1, 2, 3)) > 0)
        tflux, _, tellips = get_single_galaxy_measurements(
            noiseless_images, backgrounds, no_bar=False
        )
        eflux, _, pellips = get_single_galaxy_measurements(
            recon_no_background, backgrounds, no_bar=False
        )
        true_meas = {"true_fluxes": tflux, "true_ellips": tellips}
        recon_meas = {"recon_fluxes": eflux, "recon_ellips": pellips}
        measurements = {**true_meas, **recon_meas, "snr": snr}

        print("Saving AE results...")

        return {
            "random": {
                "true": images[rand_indices],
                "recon": recon_means[rand_indices],
                "res": residuals[rand_indices],
            },
            "worst": {
                "true": images[worst_indices],
                "recon": recon_means[worst_indices],
                "res": residuals[worst_indices],
            },
            "measurements": measurements,
        }

    def _get_binned_measurements_figure(self, data) -> Figure:
        meas = data["measurements"]
        delta_snr = 0.1

        fig, axes = plt.subplots(1, 3, figsize=(18, 7))
        ax1, ax2, ax3 = axes.flatten()
        snr = meas["snr"]
        xticks = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
        xlims = (0.5, 3)
        xlabel = r"$\log_{10} \rm SNR$"

        # fluxes
        true_fluxes, recon_fluxes = meas["true_fluxes"], meas["recon_fluxes"]
        x, y = np.log10(snr), (recon_fluxes - true_fluxes) / true_fluxes
        scatter_shade_plot(ax1, x, y, xlims, delta=delta_snr, use_boot=True)
        ax1.set_xlim(xlims)
        ax1.set_xlabel(xlabel)
        ax1.set_ylabel(r"$(f^{\rm recon} - f^{\rm true}) / f^{\rm true}$")
        ax1.set_xticks(xticks)
        ax1.axhline(0, ls="--", color="k")

        # ellipticities
        true_ellip1, recon_ellip1 = meas["true_ellips"][:, 0], meas["recon_ellips"][:, 0]
        mask1 = np.isnan(true_ellip1)  # only need first componenet for masks by construction
        mask2 = np.isnan(recon_ellip1)
        mask = ~mask1 & ~mask2
        print(f"INFO: Total number of true ellipticity NaNs is: {sum(mask1)}")
        print(f"INFO: Total number of reconstructed ellipticity NaNs is: {sum(mask2)}")
        x, y = np.log10(snr[mask]), recon_ellip1[mask] - true_ellip1[mask]
        scatter_shade_plot(ax2, x, y, xlims, delta=delta_snr, use_boot=True)
        ax2.set_xlim(xlims)
        ax2.set_xlabel(xlabel)
        ax2.set_ylabel(r"$g_{1}^{\rm recon} - g_{1}^{\rm true}$")
        ax2.set_xticks(xticks)
        ax2.axhline(0, ls="--", color="k")
        ax2.set_ylim(-0.1, 0.1)

        true_ellip2, recon_ellip2 = meas["true_ellips"][:, 1], meas["recon_ellips"][:, 1]
        x, y = np.log10(snr[mask]), recon_ellip2[mask] - true_ellip2[mask]
        scatter_shade_plot(ax3, x, y, xlims, delta=delta_snr, use_boot=True)
        ax3.set_xlim(xlims)
        ax3.set_xlabel(xlabel)
        ax3.set_ylabel(r"$g_{2}^{\rm recon} - g_{2}^{\rm true}$")
        ax3.set_xticks(xticks)
        ax3.axhline(0, ls="--", color="k")
        ax3.set_ylim(-0.1, 0.1)

        return fig

    def _get_ae_hists_figure(self, data) -> Figure:
        snr = data["measurements"]["snr"]
        fig, ax = plt.subplots(1, 1, figsize=(7, 7))
        xticks = [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
        xlims = (0, 3)
        snr_bins = np.arange(xlims[0], xlims[1] + 0.2, 0.2)
        xlabel = r"$\log_{10} \rm SNR$"
        ax.hist(np.log10(snr), bins=snr_bins, histtype="step")
        ax.set_xlabel(xlabel)
        ax.set_xticks(xticks)
        ax.axvline(np.log10(snr).mean(), label=r"\rm Mean", color="k", ls="--")

        ax.legend(prop={"size": 18})

        return fig

    def create_figure(self, fname: str, data) -> Figure:
        if fname == "ae_random_recon":
            return _reconstruction_figure(self.n_examples, *data["random"].values())
        if fname == "ae_worst_recon":
            return _reconstruction_figure(self.n_examples, *data["worst"].values())
        if fname == "ae_bin_measurements":
            return self._get_binned_measurements_figure(data)
        if fname == "ae_bin_hists":
            return self._get_ae_hists_figure(data)
        raise NotImplementedError("Figure {fname} not implemented.")


def _reconstruction_figure(
    n_examples: int, images, recons, residuals, figsize=(12, 20), hspace=-0.4
) -> Figure:
    pad = 6.0
    fig, axes = plt.subplots(nrows=n_examples, ncols=3, figsize=figsize)
    assert images.shape[0] == recons.shape[0] == residuals.shape[0] == n_examples
    assert images.shape[1] == recons.shape[1] == residuals.shape[1] == 1, "1 band only."

    # pick standard ranges for residuals
    vmin_res = residuals.min().item()
    vmax_res = residuals.max().item()

    for i in range(n_examples):
        ax_true = axes[i, 0]
        ax_recon = axes[i, 1]
        ax_res = axes[i, 2]

        # only add titles to the first axes.
        if i == 0:
            ax_true.set_title(r"\rm Images $x$", pad=pad)
            ax_recon.set_title(r"\rm Reconstruction $\tilde{x}$", pad=pad)
            ax_res.set_title(
                r"\rm Residual $\left(\tilde{x} - x\right) / \sqrt{\tilde{x}}$", pad=pad
            )

        # standarize ranges of true and reconstruction
        image = images[i, 0]
        recon = recons[i, 0]
        residual = residuals[i, 0]

        vmin = min(image.min().item(), recon.min().item())
        vmax = max(image.max().item(), recon.max().item())

        # plot images
        plot_image(fig, ax_true, image, vrange=(vmin, vmax))
        plot_image(fig, ax_recon, recon, vrange=(vmin, vmax))
        plot_image(fig, ax_res, residual, vrange=(vmin_res, vmax_res))

    plt.subplots_adjust(hspace=hspace)
    plt.tight_layout()

    return fig
