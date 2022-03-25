#!/usr/bin/env python3
"""Produce all figures. Save to PNG format."""
import warnings
from abc import abstractmethod
from pathlib import Path
from typing import Union

import galsim
import hydra
import matplotlib as mpl
import numpy as np
import pytorch_lightning as pl
import seaborn as sns
import torch
from astropy.table import Table
from astropy.wcs.wcs import WCS
from hydra.utils import instantiate
from matplotlib import pyplot as plt

from bliss import generate, reporting
from bliss.catalog import FullCatalog
from bliss.datasets import sdss
from bliss.datasets.galsim_galaxies import load_psf_from_file
from bliss.encoder import Encoder
from bliss.inference import SDSSFrame, SimulatedFrame, reconstruct_scene_at_coordinates
from bliss.models.galaxy_net import OneCenteredGalaxyAE

pl.seed_everything(42)


CB_color_cycle = [
    "#377eb8",
    "#ff7f00",
    "#4daf4a",
    "#f781bf",
    "#a65628",
    "#984ea3",
    "#999999",
    "#e41a1c",
    "#dede00",
]


def set_rc_params(
    figsize=(10, 10),
    fontsize=18,
    title_size="large",
    label_size="medium",
    legend_fontsize="medium",
    tick_label_size="small",
    major_tick_size=7,
    minor_tick_size=4,
    major_tick_width=0.8,
    minor_tick_width=0.6,
    lines_marker_size=8,
):
    # named size options: 'xx-small', 'x-small', 'small', 'medium', 'large', 'x-large', 'xx-large'.
    rc_params = {
        # font.
        "font.family": "serif",
        "font.sans-serif": "Helvetica",
        "text.usetex": True,
        "mathtext.fontset": "cm",
        "font.size": fontsize,
        # figure
        "figure.figsize": figsize,
        # axes
        "axes.labelsize": label_size,
        "axes.titlesize": title_size,
        # ticks
        "xtick.labelsize": tick_label_size,
        "ytick.labelsize": tick_label_size,
        "xtick.major.size": major_tick_size,
        "ytick.major.size": major_tick_size,
        "xtick.major.width": major_tick_width,
        "ytick.major.width": major_tick_width,
        "ytick.minor.size": minor_tick_size,
        "xtick.minor.size": minor_tick_size,
        "xtick.minor.width": minor_tick_width,
        "ytick.minor.width": minor_tick_width,
        # markers
        "lines.markersize": lines_marker_size,
        # legend
        "legend.fontsize": legend_fontsize,
        # colors
        "axes.prop_cycle": mpl.cycler(color=CB_color_cycle),
    }
    mpl.rcParams.update(rc_params)
    sns.set_context(rc=rc_params)


def format_plot(ax, xlims=None, ylims=None, xticks=None, yticks=None, xlabel="", ylabel=""):
    if xlims is not None:
        ax.set_xlim(xlims)
    if ylims is not None:
        ax.set_ylim(ylims)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if xticks:
        ax.set_xticks(xticks)
    if yticks:
        ax.set_yticks(yticks)


def remove_outliers(*args, level=0.99):
    # each arg in args should be 1D numpy.array with same number of data points.
    for arg in args:
        assert len(arg) == len(args[0])
        assert isinstance(arg, np.ndarray)
        assert len(arg.shape) == 1
    keep = np.ones(args[0].shape).astype(bool)
    for x in args:
        x_min, x_max = np.quantile(x, 1 - level), np.quantile(x, level)
        keep_x = (x > x_min) & (x < x_max)
        keep &= keep_x

    return (arg[keep] for arg in args)


def add_extra_coadd_info(coadd_cat_file: str, psf_image_file: str, pixel_scale: float, wcs: WCS):
    """Add additional useful information to coadd catalog."""
    coadd_cat = Table.read(coadd_cat_file)

    psf = load_psf_from_file(psf_image_file, pixel_scale)
    x, y = wcs.all_world2pix(coadd_cat["ra"], coadd_cat["dec"], 0)
    galaxy_bools = ~coadd_cat["probpsf"].data.astype(bool)
    flux, mag = reporting.get_flux_coadd(coadd_cat)
    hlr = reporting.get_hlr_coadd(coadd_cat, psf)

    coadd_cat["x"] = x
    coadd_cat["y"] = y
    coadd_cat["galaxy_bool"] = galaxy_bools
    coadd_cat["flux"] = flux
    coadd_cat["mag"] = mag
    coadd_cat["hlr"] = hlr
    coadd_cat.replace_column("is_saturated", coadd_cat["is_saturated"].data.astype(bool))
    coadd_cat.write(coadd_cat_file, overwrite=True)  # overwrite with additional info.


class BlissFigures:
    def __init__(self, outdir, cache="temp.pt", overwrite=False) -> None:

        self.outdir = Path(outdir)
        self.cache = self.outdir / cache
        self.overwrite = overwrite

    @property
    @abstractmethod
    def fignames(self):
        """What figures will be produced with this class? What are their names?"""

    def get_data(self, *args, **kwargs):
        """Return summary of data for producing plot, must be cachable w/ torch.save()."""
        if self.cache.exists() and not self.overwrite:
            return torch.load(self.cache)

        data = self.compute_data(*args, **kwargs)
        torch.save(data, self.cache)
        return data

    @abstractmethod
    def compute_data(self, *args, **kwargs):
        return {}

    def save_figures(self, *args, **kwargs):
        """Create figures and save to output directory with names from `self.fignames`."""
        data = self.get_data(*args, **kwargs)
        figs = self.create_figures(data)
        for k, fname in self.fignames.items():
            figs[k].savefig(self.outdir / fname, format="png")

    @abstractmethod
    def create_figures(self, data):
        """Return matplotlib figure instances to save based on data."""
        return mpl.figure.Figure()


class AEReconstructionFigures(BlissFigures):
    def __init__(self, outdir="", cache="ae_cache.pt", overwrite=False, n_examples=5) -> None:
        super().__init__(outdir=outdir, cache=cache, overwrite=overwrite)
        self.n_examples = n_examples

    @property
    def fignames(self):
        return {
            "random_recon": "random_reconstructions.png",
            "worst_recon": "worst_reconstructions.png",
            "measure_contours": "single_galaxy_measurements_contours.png",
            "measure_scatter_bins": "single_galaxy_scatter_bins.png",
        }

    def compute_data(
        self, autoencoder: OneCenteredGalaxyAE, images_file, psf_file, sdss_pixel_scale
    ):
        # NOTE: For very dim objects (max_pixel < 6, flux ~ 1000), autoencoder can return
        # 0.1% objects with negative flux. This objects are discarded.
        device = autoencoder.device  # GPU is better otherwise slow.

        image_data = torch.load(images_file)
        images = image_data["images"]
        recon_means = torch.tensor([])
        background = image_data["background"].reshape(1, 1, 53, 53).to(device)
        noiseless_images = image_data["noiseless"].numpy()  # no background or noise.

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
        psf_image = psf.drawImage(nx=53, ny=53, scale=sdss_pixel_scale).array

        recon_no_background = recon_means.numpy() - background.cpu().numpy()
        # a small percentage of low magnitude objects end up predicted with negative flux.
        good_bool = recon_no_background.sum(axis=(1, 2, 3)) > 0
        measurements = reporting.get_single_galaxy_measurements(
            slen=53,
            true_images=noiseless_images[good_bool],
            recon_images=recon_no_background[good_bool],
            psf_image=psf_image.reshape(-1, 53, 53),
            pixel_scale=sdss_pixel_scale,
        )

        return {
            "random": (images[rand_indices], recon_means[rand_indices], residuals[rand_indices]),
            "worst": (images[worst_indices], recon_means[worst_indices], residuals[worst_indices]),
            "measurements": measurements,
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
            image = images[i, 0].detach().cpu().numpy()
            recon = recons[i, 0].detach().cpu().numpy()
            residual = residuals[i, 0].detach().cpu().numpy()

            vmin = min(image.min().item(), recon.min().item())
            vmax = max(image.max().item(), recon.max().item())

            # plot images
            reporting.plot_image(fig, ax_true, image, vrange=(vmin, vmax))
            reporting.plot_image(fig, ax_recon, recon, vrange=(vmin, vmax))
            reporting.plot_image(fig, ax_res, residual, vrange=(vmin_res, vmax_res))

        plt.subplots_adjust(hspace=-0.4)
        plt.tight_layout()

        return fig

    def make_scatter_contours(self, ax, x, y, **plot_kwargs):
        sns.scatterplot(x=x, y=y, s=10, color="0.15", ax=ax)
        sns.histplot(x=x, y=y, pthresh=0.1, cmap="mako", ax=ax, cbar=True)
        sns.kdeplot(x=x, y=y, levels=10, color="w", linewidths=1, ax=ax)
        format_plot(ax, **plot_kwargs)

    def make_twod_hist(self, x, y, color="m", height=7, **plot_kwargs):
        # NOTE: This creates its own figure object which makes it hard to use.
        # TODO: Revive if useful later on.
        g = sns.jointplot(x=x, y=y, color=color, kind="hist", marginal_ticks=True, height=height)
        g.ax_joint.axline(xy1=(np.median(x), np.median(y)), slope=1.0)
        format_plot(g.ax_joint, **plot_kwargs)

    def scatter_bin_plot(self, ax, x, y, xlims, delta, capsize=5.0, **plot_kwargs):

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

    def make_scatter_contours_plot(self, meas):
        sns.set_theme(style="darkgrid")
        set_rc_params(
            fontsize=22, legend_fontsize="small", tick_label_size="small", label_size="medium"
        )
        fig, axes = plt.subplots(2, 2, figsize=(12, 12))
        ax1, ax2, ax3, ax4 = axes.flatten()

        # fluxes / magnitudes
        x, y = remove_outliers(meas["true_mags"], meas["recon_mags"], level=0.95)
        mag_ticks = (16, 17, 18, 19)
        xlabel = r"$m^{\rm true}$"
        ylabel = r"$m^{\rm recon}$"
        self.make_scatter_contours(
            ax1, x, y, xlabel=xlabel, ylabel=ylabel, xticks=mag_ticks, yticks=mag_ticks
        )

        # hlrs
        x, y = remove_outliers(meas["true_hlrs"], meas["recon_hlrs"], level=0.95)
        self.make_scatter_contours(ax2, x, y, xlabel=r"$r^{\rm true}$", ylabel=r"$r^{\rm recon}$")

        # ellipticities 1
        x, y = remove_outliers(meas["true_ellip"][:, 0], meas["recon_ellip"][:, 0], level=0.95)
        g_ticks = (-1.0, -0.5, 0.0, 0.5, 1.0)
        xlabel = r"$g_{1}^{\rm true}$"
        ylabel = r"$g_{1}^{\rm recon}$"
        self.make_scatter_contours(
            ax3, x, y, xticks=g_ticks, yticks=g_ticks, xlabel=xlabel, ylabel=ylabel
        )

        # ellipticities 2
        x, y = remove_outliers(meas["true_ellip"][:, 1], meas["recon_ellip"][:, 1], level=0.95)
        xlabel = r"$g_{2}^{\rm true}$"
        ylabel = r"$g_{2}^{\rm recon}$"
        self.make_scatter_contours(
            ax4, x, y, xticks=g_ticks, yticks=g_ticks, xlabel=xlabel, ylabel=ylabel
        )

        plt.tight_layout()
        return fig

    def make_scatter_bin_plots(self, meas):
        fig, axes = plt.subplots(2, 2, figsize=(16, 16))
        ax1, ax2, ax3, ax4 = axes.flatten()
        set_rc_params(fontsize=24)

        # fluxes / magnitudes
        true_mags, recon_mags = meas["true_mags"], meas["recon_mags"]
        x, y = remove_outliers(true_mags, (recon_mags - true_mags) / true_mags, level=0.99)
        self.scatter_bin_plot(
            ax1,
            x,
            y,
            xlims=(x.min(), x.max()),
            delta=0.25,
            xlabel=r"\rm $m^{\rm true}$",
            ylabel=r"\rm $(m^{\rm recon} - m^{\rm true}) / m^{\rm true}$",
            xticks=[16, 17, 18, 19, 20],
        )

        # hlrs
        true_hlrs, recon_hlrs = meas["true_hlrs"], meas["recon_hlrs"]
        x, y = remove_outliers(true_hlrs, (recon_hlrs - true_hlrs) / true_hlrs, level=0.99)
        self.scatter_bin_plot(
            ax2,
            x,
            y,
            xlims=(x.min(), x.max()),
            delta=0.5,
            xlabel=r"$r^{\rm true}$",
            ylabel=r"$(r^{\rm recon} - r^{\rm true}) / r^{\rm true}$",
        )

        # ellipticities
        true_ellip1, recon_ellip1 = meas["true_ellip"][:, 0], meas["recon_ellip"][:, 0]
        x, y = remove_outliers(true_ellip1, recon_ellip1 - true_ellip1, level=0.99)
        self.scatter_bin_plot(
            ax3,
            x,
            y,
            xlims=(-0.85, 0.85),
            delta=0.2,
            xticks=[-1.0, -0.5, 0.0, 0.5, 1.0],
            xlabel=r"$g_{1}^{\rm true}$",
            ylabel=r"$g_{1}^{\rm recon} - g_{1}^{\rm true}$",
        )

        true_ellip2, recon_ellip2 = meas["true_ellip"][:, 1], meas["recon_ellip"][:, 1]
        x, y = remove_outliers(true_ellip2, recon_ellip2 - true_ellip2, level=0.99)
        self.scatter_bin_plot(
            ax4,
            x,
            y,
            xlims=(-0.85, 0.85),
            delta=0.2,
            xticks=[-1.0, -0.5, 0.0, 0.5, 1.0],
            xlabel=r"$g_{2}^{\rm true}$",
            ylabel=r"$g_{2}^{\rm recon} - g_{2}^{\rm true}$",
        )

        plt.tight_layout()

        return fig

    def create_figures(self, data):

        return {
            "random_recon": self.reconstruction_figure(*data["random"]),
            "worst_recon": self.reconstruction_figure(*data["worst"]),
            "measure_contours": self.make_scatter_contours_plot(data["measurements"]),
            "measure_scatter_bins": self.make_scatter_bin_plots(data["measurements"]),
        }


class DetectionClassificationFigures(BlissFigures):
    def __init__(self, outdir="", cache="detect_class.pt", overwrite=False) -> None:
        super().__init__(outdir=outdir, cache=cache, overwrite=overwrite)

    @property
    def fignames(self):
        return {
            "detection": "sdss-precision-recall.png",
            "classification": "sdss-classification-acc.png",
            "scatter": "sdss-mag-class-scatter.png",
            "flux_comparison": "flux-comparison.png",
        }

    def compute_data(self, frame: Union[SDSSFrame, SimulatedFrame], encoder, decoder):
        bp = encoder.border_padding
        device = encoder.device
        slen = 300  # chunk side-length for whole iamge.
        h, w = bp, bp
        h_end = ((frame.image.shape[2] - 2 * bp) // 4) * 4 + bp
        w_end = ((frame.image.shape[3] - 2 * bp) // 4) * 4 + bp
        coadd_params: FullCatalog = frame.get_catalog((h, h_end), (w, w_end))

        # misclassified galaxies in PHOTO as galaxies (obtaind by eye)
        ids = [8647475119820964111, 8647475119820964100, 8647475119820964192]
        for my_id in ids:
            idx = np.where(coadd_params["objid"] == my_id)[0].item()
            coadd_params["galaxy_bools"][:, idx, :] = 0

        _, tile_est_params = reconstruct_scene_at_coordinates(
            encoder,
            decoder,
            frame.image,
            frame.background,
            h_range=(h, h_end),
            w_range=(w, w_end),
            slen=slen,
            device=device,
        )
        est_params = tile_est_params.to_full_params()
        est_params.plocs -= 0.5
        est_params["fluxes"] = (
            est_params["galaxy_bools"] * est_params["galaxy_fluxes"]
            + est_params["star_bools"] * est_params["fluxes"]
        )

        est_params["mags"] = sdss.convert_flux_to_mag(est_params["fluxes"])

        mag_bins = np.arange(18, 23, 0.25)  # skip 23
        precisions = []
        recalls = []
        class_accs = []
        galaxy_accs = []
        star_accs = []

        # compute data for precision/recall/classification accuracy as a function of magnitude.
        for mag in mag_bins:
            res = reporting.scene_metrics(
                coadd_params, est_params, mag_cut=mag, slack=1.0, mag_slack=1.0
            )
            precisions.append(res["precision"].item())
            recalls.append(res["recall"].item())
            class_accs.append(res["class_acc"].item())

            # how many out of the matched galaxies are accurately classified?
            galaxy_acc = res["conf_matrix"][0, 0] / res["conf_matrix"][0, :].sum()
            galaxy_accs.append(galaxy_acc)

            # how many out of the matched stars are correctly classified?
            star_acc = res["conf_matrix"][1, 1] / res["conf_matrix"][1, :].sum()
            star_accs.append(star_acc)

        # data for scatter plot of misclassifications.
        tplocs = coadd_params.plocs.reshape(-1, 2)
        eplocs = est_params.plocs.reshape(-1, 2)
        tindx, eindx, dkeep, _ = reporting.match_by_locs(tplocs, eplocs, slack=1.0)
        tgbool = coadd_params["galaxy_bools"].reshape(-1)[tindx][dkeep]
        egbool = est_params["galaxy_bools"].reshape(-1)[eindx][dkeep]
        egprob = est_params["galaxy_probs"].reshape(-1)[eindx][dkeep]
        tmag = coadd_params["mags"].reshape(-1)[tindx][dkeep]
        emag = est_params["mags"].reshape(-1)[eindx][dkeep]

        return {
            "mag_bins": mag_bins,
            "precisions": precisions,
            "recalls": recalls,
            "class_accs": class_accs,
            "star_accs": star_accs,
            "galaxy_accs": galaxy_accs,
            "scatter_class": (tgbool, egbool, egprob, tmag, emag),
        }

    def create_figures(self, data):  # pylint: disable=too-many-statements
        """Make figures related to detection and classification in SDSS."""
        sns.set_theme(style="darkgrid")

        mag_bins = data["mag_bins"]
        recalls = data["recalls"]
        precisions = data["precisions"]
        class_accs = data["class_accs"]
        galaxy_accs = data["galaxy_accs"]
        star_accs = data["star_accs"]

        # precision / recall
        set_rc_params(tick_label_size=22, label_size=30)
        f1, ax = plt.subplots(1, 1, figsize=(10, 10))
        format_plot(ax, xlabel=r"\rm magnitude cut", ylabel="value of metric")
        ax.plot(mag_bins, recalls, "-o", label=r"\rm recall")
        ax.plot(mag_bins, precisions, "-o", label=r"\rm precision")
        plt.xlim(18, 23)
        ax.legend(loc="best", prop={"size": 22})

        # classification accuracy
        set_rc_params(tick_label_size=22, label_size=30)
        f2, ax = plt.subplots(1, 1, figsize=(10, 10))
        format_plot(ax, xlabel=r"\rm magnitude cut", ylabel="accuracy")
        ax.plot(mag_bins, class_accs, "-o", label=r"\rm classification accuracy")
        ax.plot(mag_bins, galaxy_accs, "-o", label=r"\rm galaxy classification accuracy")
        ax.plot(mag_bins, star_accs, "-o", label=r"\rm star classification accuracy")
        plt.xlim(18, 23)
        ax.legend(loc="best", prop={"size": 22})

        # magnitude / classification scatter
        set_rc_params(tick_label_size=22, label_size=30)
        f3, ax = plt.subplots(1, 1, figsize=(10, 10))
        tgbool, egbool, egprob, tmag, emag = data["scatter_class"]
        tgbool, egbool = tgbool.numpy().astype(bool), egbool.numpy().astype(bool)
        egprob = egprob.numpy()
        tmag, emag = tmag.numpy(), emag.numpy()
        correct = np.equal(tgbool, egbool).astype(bool)
        ax.scatter(tmag[correct], egprob[correct], marker="+", c="b", label="correct", alpha=0.5)
        ax.scatter(
            tmag[~correct], egprob[~correct], marker="x", c="r", label="incorrect", alpha=0.5
        )
        ax.axhline(0.5, linestyle="--")
        ax.axhline(0.1, linestyle="--")
        ax.axhline(0.9, linestyle="--")
        ax.set_xlabel("True Magnitude")
        ax.set_ylabel("Estimated Probability of Galaxy")
        ax.legend(loc="best", prop={"size": 22})

        # mag / mag scatter
        f4, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 7))
        ax1.scatter(tmag[tgbool], emag[tgbool], marker="o", c="r", alpha=0.5)
        ax1.plot([15, 23], [15, 23], c="r", label="x=y line")
        ax2.scatter(tmag[~tgbool], emag[~tgbool], marker="o", c="b", alpha=0.5)
        ax2.plot([15, 23], [15, 23], c="b", label="x=y line")
        ax1.legend(loc="best", prop={"size": 22})
        ax2.legend(loc="best", prop={"size": 22})

        ax1.set_xlabel("True Magnitude")
        ax2.set_xlabel("True Magnitude")
        ax1.set_ylabel("Estimated Magnitude")
        ax2.set_ylabel("Estimated Magnitude")
        ax1.set_title("Matched Coadd Galaxies")
        ax2.set_title("Matched Coadd Stars")
        return {"detection": f1, "classification": f2, "scatter": f3, "flux_comparison": f4}


class SDSSReconstructionFigures(BlissFigures):
    def __init__(self, scenes, outdir="", cache="recon_sdss.pt", overwrite=False) -> None:
        self.scenes = scenes
        super().__init__(outdir=outdir, cache=cache, overwrite=overwrite)

    @property
    def fignames(self):
        return {k: k + ".png" for k in self.scenes}

    def compute_data(self, frame: Union[SDSSFrame, SimulatedFrame], encoder, decoder):

        device = encoder.device
        data = {}

        for figname, scene_coords in self.scenes.items():
            h, w, scene_size = scene_coords["h"], scene_coords["w"], scene_coords["size"]
            assert scene_size <= 300, "Scene too large, change slen."
            h_end = h + scene_size
            w_end = w + scene_size
            true = frame.image[:, :, h:h_end, w:w_end]
            coadd_params = frame.get_catalog((h, h_end), (w, w_end))

            recon, tile_map_recon = reconstruct_scene_at_coordinates(
                encoder,
                decoder,
                frame.image,
                frame.background,
                h_range=(h, h_end),
                w_range=(w, w_end),
                slen=scene_size,
                device=device,
            )
            resid = (true - recon) / recon.sqrt()

            recon_map = tile_map_recon.to_full_params()
            recon_map.plocs = recon_map.plocs - 0.5

            true = true[0, 0].cpu().numpy()
            recon = recon[0, 0].cpu().numpy()
            resid = resid[0, 0].cpu().numpy()
            data[figname] = (true, recon, resid, coadd_params, recon_map)

        return data

    def create_figures(self, data):  # pylint: disable=too-many-statements
        """Make figures related to reconstruction in SDSS."""
        out_figures = {}

        pad = 6.0
        sns.set_style("white")
        set_rc_params(fontsize=22, tick_label_size="small", legend_fontsize="small")
        for figname, scene_coords in self.scenes.items():
            scene_size = scene_coords["size"]
            true, recon, res, coadd_params, recon_map = data[figname]
            fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(28, 12))
            assert len(true.shape) == len(recon.shape) == len(res.shape) == 2

            ax_true = axes[0]
            ax_recon = axes[1]
            ax_res = axes[2]

            ax_true.set_title("Original Image", pad=pad)
            ax_recon.set_title("Reconstruction", pad=pad)
            ax_res.set_title("Residual", pad=pad)

            # plot images
            reporting.plot_image(fig, ax_true, true, vrange=(800, 1000))
            reporting.plot_image(fig, ax_recon, recon, vrange=(800, 1000))
            reporting.plot_image(fig, ax_res, res, vrange=(-5, 5))

            locs_true = coadd_params.plocs.reshape(-1, 2)
            true_galaxy_bools = coadd_params["galaxy_bools"].reshape(-1).bool()
            locs_galaxies_true = locs_true[true_galaxy_bools]
            locs_stars_true = locs_true[~true_galaxy_bools]

            s = 55 * 300 / scene_size  # marker size
            lw = 2 * np.sqrt(300 / scene_size)

            if locs_stars_true.shape[0] > 0:
                x, y = locs_stars_true[:, 1], locs_stars_true[:, 0]
                ax_true.scatter(x, y, color="blue", marker="+", s=s, linewidths=lw)
                ax_recon.scatter(
                    x, y, color="blue", marker="+", s=s, label="SDSS Stars", linewidths=lw
                )
                ax_res.scatter(x, y, color="blue", marker="+", s=s, linewidths=lw, alpha=0.5)
            if locs_galaxies_true.shape[0] > 0:
                x, y = locs_galaxies_true[:, 1], locs_galaxies_true[:, 0]
                ax_true.scatter(x, y, color="red", marker="+", s=s, linewidths=lw)
                ax_recon.scatter(
                    x, y, color="red", marker="+", s=s, label="SDSS Galaxies", linewidths=lw
                )
                ax_res.scatter(x, y, color="red", marker="+", s=s, linewidths=lw, alpha=0.5)

            if recon_map is not None:
                s *= 0.75
                lw *= 0.75
                locs_pred = recon_map.plocs.reshape(-1, 2)
                star_bools = recon_map["star_bools"].reshape(-1).bool()
                galaxy_bools = recon_map["galaxy_bools"].reshape(-1).bool()
                locs_galaxies = locs_pred[galaxy_bools]
                locs_stars = locs_pred[star_bools]
                if locs_stars.shape[0] > 0:
                    label = "Predicted Star"
                    in_bounds = torch.all((locs_stars > 0) & (locs_stars < scene_size), dim=-1)
                    locs_stars = locs_stars[in_bounds]
                    x, y = locs_stars[:, 1], locs_stars[:, 0]
                    ax_true.scatter(x, y, color="aqua", marker="x", s=s, linewidths=lw)
                    ax_recon.scatter(
                        x, y, color="aqua", marker="x", s=s, label=label, linewidths=lw
                    )
                    ax_res.scatter(x, y, color="aqua", marker="x", s=s, linewidths=lw, alpha=0.5)

                if locs_galaxies.shape[0] > 0:
                    label = "Predicted Galaxy"
                    in_bounds = torch.all(
                        (locs_galaxies > 0) & (locs_galaxies < scene_size), dim=-1
                    )
                    locs_galaxies = locs_galaxies[in_bounds]
                    x, y = locs_galaxies[:, 1], locs_galaxies[:, 0]
                    ax_true.scatter(x, y, color="hotpink", marker="x", s=s, linewidths=lw)
                    ax_recon.scatter(
                        x, y, color="hotpink", marker="x", s=s, label=label, linewidths=lw
                    )
                    ax_res.scatter(x, y, color="hotpink", marker="x", s=s, linewidths=lw, alpha=0.5)
                ax_recon.legend(
                    bbox_to_anchor=(0.0, 1.2, 1.0, 0.102),
                    loc="lower left",
                    ncol=2,
                    mode="expand",
                    borderaxespad=0.0,
                )

            plt.subplots_adjust(hspace=-0.4)
            plt.tight_layout()

            out_figures[figname] = fig

        return out_figures


@hydra.main(config_path="./config", config_name="config")
def plots(cfg):  # pylint: disable=too-many-statements

    figs = set(cfg.plots.figs)
    outdir = cfg.plots.outdir
    overwrite = cfg.plots.overwrite
    device = torch.device(cfg.plots.device)

    if not Path(outdir).exists():
        warnings.warn("Specified output directory does not exist, will attempt to create it.")
        Path(outdir).mkdir(exist_ok=True, parents=True)

    if figs.intersection({2, 3}):
        # load SDSS frame
        frame: Union[SDSSFrame, SimulatedFrame] = instantiate(cfg.plots.frame)

        # load models required for SDSS reconstructions.
        sleep = instantiate(cfg.models.sleep)
        sleep.load_state_dict(torch.load(cfg.predict.sleep_checkpoint))
        location = sleep.image_encoder.to(device).eval()

        binary = instantiate(cfg.models.binary)
        binary.load_state_dict(torch.load(cfg.predict.binary_checkpoint))
        binary = binary.to(device).eval()

        galaxy = instantiate(cfg.models.galaxy_encoder)
        galaxy.load_state_dict(torch.load(cfg.predict.galaxy_checkpoint))
        galaxy = galaxy.to(device).eval()

        decoder = sleep.image_decoder.to(device).eval()
        encoder = Encoder(location.eval(), binary.eval(), galaxy.eval()).to(device)

    # FIGURE 1: Autoencoder single galaxy reconstruction
    if 1 in figs:
        print("INFO: Creating autoencoder figures...")
        autoencoder = instantiate(cfg.models.galaxy_net)
        autoencoder.load_state_dict(torch.load(cfg.models.prior.galaxy_prior.autoencoder_ckpt))
        autoencoder = autoencoder.to(device).eval()

        # generate galsim simulated galaxies images if file does not exist.
        galaxies_file = Path(cfg.plots.simulated_sdss_individual_galaxies)
        if not galaxies_file.exists():
            print(f"INFO: Generating individual galaxy images and saving to: {galaxies_file}")
            dataset = instantiate(
                cfg.datasets.sdss_galaxies, batch_size=512, n_batches=20, num_workers=20
            )
            imagepath = galaxies_file.parent / (galaxies_file.stem + "_images.jpg")
            generate.generate(
                dataset, galaxies_file, imagepath, n_plots=25, global_params=("background", "slen")
            )

        # create figure classes and plot.
        ae_figures = AEReconstructionFigures(outdir, overwrite=overwrite, n_examples=5)
        ae_figures.save_figures(
            autoencoder, galaxies_file, cfg.plots.psf_file, cfg.plots.sdss_pixel_scale
        )
        mpl.rc_file_defaults()

    # FIGURE 2: Classification and Detection metrics
    if 2 in figs:
        print("INFO: Creating classification and detection metrics from SDSS frame figures...")
        dc_fig = DetectionClassificationFigures(outdir, overwrite=overwrite)
        dc_fig.save_figures(frame, encoder, decoder)
        mpl.rc_file_defaults()

    # FIGURE 3: Reconstructions on SDSS
    if 3 in figs:
        print("INFO: Creating reconstructions from SDSS figures...")
        sdss_rec_fig = SDSSReconstructionFigures(cfg.plots.scenes, outdir, overwrite=overwrite)
        sdss_rec_fig.save_figures(frame, encoder, decoder)
        mpl.rc_file_defaults()

    if not figs.intersection({1, 2, 3}):
        raise NotImplementedError(
            "No figures were created, `cfg.plots.figs` should be a subset of [1,2,3]."
        )


if __name__ == "__main__":
    plots()  # pylint: disable=no-value-for-parameter
