import galsim
import numpy as np
import seaborn as sns
import torch
from matplotlib import pyplot as plt
from torch import Tensor

from bliss import reporting
from bliss.models.galaxy_net import OneCenteredGalaxyAE
from case_studies.sdss_galaxies.plots.bliss_figures import BlissFigures, format_plot, set_rc_params


def scatter_bin_plot(ax, x, y, xlims, delta, capsize=5.0, **plot_kwargs):
    # plot median and 25/75 quantiles on each bin decided by delta and xlims.

    xbins = np.arange(xlims[0], xlims[1], delta)

    xs = np.zeros(len(xbins))
    ys = np.zeros(len(xbins))
    errs = np.zeros((len(xbins), 2))

    for i, bx in enumerate(xbins):
        keep_x = (x > bx) & (x < bx + delta)
        y_bin = y[keep_x]

        xs[i] = bx + delta / 2

        if len(y_bin) == 0:  # noqa: WPS507
            ys[i] = np.nan
            errs[i] = (np.nan, np.nan)
            continue

        ys[i] = np.median(y_bin)
        errs[i, :] = ys[i] - np.quantile(y_bin, 0.25), np.quantile(y_bin, 0.75) - ys[i]

    errs = errs.T.reshape(2, -1)
    ax.errorbar(xs, ys, yerr=errs, marker="o", c="m", linestyle="--", capsize=capsize)
    format_plot(ax, **plot_kwargs)


def make_scatter_contours(ax, x, y, **plot_kwargs):
    sns.scatterplot(x=x, y=y, s=10, color="0.15", ax=ax)
    sns.histplot(x=x, y=y, pthresh=0.1, cmap="mako", ax=ax, cbar=True)
    sns.kdeplot(x=x, y=y, levels=10, color="w", linewidths=1, ax=ax)
    format_plot(ax, **plot_kwargs)


class AEReconstructionFigures(BlissFigures):
    cache = "ae_cache.pt"

    def __init__(self, figdir, cachedir, overwrite=False, n_examples=5, img_format="png") -> None:
        super().__init__(
            figdir=figdir, cachedir=cachedir, overwrite=overwrite, img_format=img_format
        )
        self.n_examples = n_examples

    def compute_data(
        self, autoencoder: OneCenteredGalaxyAE, images_file, psf_file, sdss_pixel_scale
    ):
        # NOTE: For very dim objects (max_pixel < 6, flux ~ 1000), autoencoder can return
        # 0.1% objects with negative flux. This objects are discarded.
        device = autoencoder.device  # GPU is better otherwise slow.

        image_data = torch.load(images_file)
        true_params: Tensor = image_data["params"]
        images: Tensor = image_data["images"]
        recon_means = torch.tensor([])
        background: Tensor = image_data["background"].reshape(1, 1, 53, 53).to(device)
        noiseless_images: Tensor = image_data["noiseless"]
        snr: Tensor = image_data["snr"].reshape(-1)

        print("INFO: Computing reconstructions from saved autoencoder model...")
        n_images = images.shape[0]
        batch_size = 128
        n_iters = int(np.ceil(n_images // 128))
        for i in range(n_iters):  # in batches otherwise GPU error.
            bimages = images[batch_size * i : batch_size * (i + 1)].to(device)
            recon_mean, _ = autoencoder.forward(bimages, background)
            recon_mean = recon_mean.detach().to("cpu")
            recon_means = torch.cat((recon_means, recon_mean))
        residuals = (images - recon_means) / recon_means.sqrt()
        assert recon_means.shape[0] == noiseless_images.shape[0]

        # random
        rand_indices = torch.randint(0, len(images), size=(self.n_examples,))

        # worst
        absolute_residual = residuals.abs().sum(axis=(1, 2, 3))
        worst_indices = absolute_residual.argsort()[-self.n_examples :]

        # measurements
        psf_array = np.load(psf_file)
        galsim_psf_image = galsim.Image(psf_array[0], scale=sdss_pixel_scale)  # sdss scale
        psf = galsim.InterpolatedImage(galsim_psf_image).withFlux(1.0)
        psf_image = torch.tensor(psf.drawImage(nx=53, ny=53, scale=sdss_pixel_scale).array)

        recon_no_background = recon_means - background.cpu()
        assert torch.all(recon_no_background.sum(axis=(1, 2, 3)) > 0)
        true_meas = reporting.get_single_galaxy_measurements(
            noiseless_images, psf_image.reshape(-1, 53, 53), sdss_pixel_scale
        )
        true_meas = {f"true_{k}": v for k, v in true_meas.items()}
        recon_meas = reporting.get_single_galaxy_measurements(
            recon_no_background, psf_image.reshape(-1, 53, 53), sdss_pixel_scale
        )
        recon_meas = {f"recon_{k}": v for k, v in recon_meas.items()}
        measurements = {**true_meas, **recon_meas, "snr": snr}

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
            "true_params": true_params,
        }

    def reconstruction_figure(self, images, recons, residuals):
        pad = 6.0
        set_rc_params(fontsize=22, tick_label_size="small", legend_fontsize="small")
        fig, axes = plt.subplots(nrows=self.n_examples, ncols=3, figsize=(12, 20))

        assert images.shape[0] == recons.shape[0] == residuals.shape[0] == self.n_examples
        assert images.shape[1] == recons.shape[1] == residuals.shape[1] == 1, "1 band only."

        # pick standard ranges for residuals
        vmin_res = residuals.min().item()
        vmax_res = residuals.max().item()

        for i in range(self.n_examples):

            ax_true = axes[i, 0]
            ax_recon = axes[i, 1]
            ax_res = axes[i, 2]

            # only add titles to the first axes.
            if i == 0:
                ax_true.set_title("Images $x$", pad=pad)
                ax_recon.set_title(r"Reconstruction $\tilde{x}$", pad=pad)
                ax_res.set_title(
                    r"Residual $\left(x - \tilde{x}\right) / \sqrt{\tilde{x}}$", pad=pad
                )

            # standarize ranges of true and reconstruction
            image = images[i, 0]
            recon = recons[i, 0]
            residual = residuals[i, 0]

            vmin = min(image.min().item(), recon.min().item())
            vmax = max(image.max().item(), recon.max().item())

            # plot images
            reporting.plot_image(fig, ax_true, image, vrange=(vmin, vmax))
            reporting.plot_image(fig, ax_recon, recon, vrange=(vmin, vmax))
            reporting.plot_image(fig, ax_res, residual, vrange=(vmin_res, vmax_res))

        plt.subplots_adjust(hspace=-0.4)
        plt.tight_layout()

        return fig

    def make_scatter_contours_plot(self, meas):
        sns.set_theme(style="darkgrid")
        set_rc_params(
            fontsize=22, legend_fontsize="small", tick_label_size="small", label_size="medium"
        )
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))

        # magnitudes
        x, y = meas["true_mags"], meas["recon_mags"]
        mag_ticks = (15, 16, 17, 18, 19, 20, 21, 22, 23)
        xlabel = r"$m^{\rm true}$"
        ylabel = r"$m^{\rm recon}$"
        make_scatter_contours(
            ax,
            x,
            y,
            xlabel=xlabel,
            ylabel=ylabel,
            xticks=mag_ticks,
            yticks=mag_ticks,
            xlims=(15, 24),
            ylims=(15, 24),
        )
        ax.plot([15, 24], [15, 24], color="r", lw=2)
        plt.tight_layout()
        return fig

    def make_scatter_bin_plots(self, meas):
        fig, axes = plt.subplots(2, 2, figsize=(16, 16))
        ax1, ax2, ax3, ax4 = axes.flatten()
        set_rc_params(fontsize=24)
        snr = meas["snr"]
        xticks = [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
        xlims = (0, 3)
        xlabel = r"$\log_{10} \text{SNR}$"

        # magnitudes
        true_mags, recon_mags = meas["true_mags"], meas["recon_mags"]
        x, y = np.log10(snr), recon_mags - true_mags
        scatter_bin_plot(
            ax1,
            x,
            y,
            delta=0.2,
            xlims=xlims,
            xlabel=xlabel,
            ylabel=r"\rm $m^{\rm recon} - m^{\rm true}$",
            xticks=xticks,
        )

        # fluxes
        true_fluxes, recon_fluxes = meas["true_fluxes"], meas["recon_fluxes"]
        x, y = np.log10(snr), (recon_fluxes - true_fluxes) / recon_fluxes
        scatter_bin_plot(
            ax2,
            x,
            y,
            delta=0.2,
            xlims=xlims,
            xlabel=xlabel,
            ylabel=r"\rm $(f^{\rm recon} - f^{\rm true}) / f^{\rm recon}$",
            xticks=xticks,
        )

        # ellipticities
        true_ellip1, recon_ellip1 = meas["true_ellips"][:, 0], meas["recon_ellips"][:, 0]
        x, y = np.log10(snr), recon_ellip1 - true_ellip1
        scatter_bin_plot(
            ax3,
            x,
            y,
            delta=0.2,
            xlims=xlims,
            xticks=xticks,
            xlabel=xlabel,
            ylabel=r"$g_{1}^{\rm recon} - g_{1}^{\rm true}$",
        )

        true_ellip2, recon_ellip2 = meas["true_ellips"][:, 1], meas["recon_ellips"][:, 1]
        x, y = np.log10(snr), recon_ellip2 - true_ellip2
        scatter_bin_plot(
            ax4,
            x,
            y,
            delta=0.2,
            xlims=xlims,
            xticks=xticks,
            xlabel=xlabel,
            ylabel=r"$g_{2}^{\rm recon} - g_{2}^{\rm true}$",
        )

        plt.tight_layout()

        return fig

    def create_flux_vs_snr_plot(self, meas):
        xticks = [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
        xlims = (0, 3)
        snr, fluxes = meas["snr"], meas["true_fluxes"]
        x, y = np.log10(snr), np.log10(fluxes)
        fig, ax = plt.subplots(1, 1, figsize=(7, 7))
        make_scatter_contours(
            ax,
            x,
            y,
            xticks=xticks,
            xlims=xlims,
            ylims=(0, 7),
            xlabel=r"$\log_{10} \rm SNR$",
            ylabel=r"$\log_{10} f^{\rm true}$",
        )
        return fig

    def create_figures(self, data):
        return {
            "random_recon": self.reconstruction_figure(*data["random"].values()),
            "worst_recon": self.reconstruction_figure(*data["worst"].values()),
            "single_galaxy_meas_contours": self.make_scatter_contours_plot(data["measurements"]),
            "single_galaxy_meas_bins": self.make_scatter_bin_plots(data["measurements"]),
            "flux_vs_snr": self.create_flux_vs_snr_plot(data["measurements"]),
        }
