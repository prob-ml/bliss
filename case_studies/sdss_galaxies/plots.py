#!/usr/bin/env python3
"""Produce all figures. Save to PNG format."""
import warnings
from abc import abstractmethod
from collections import defaultdict
from pathlib import Path
from typing import Union

import galsim
import hydra
import matplotlib as mpl
import numpy as np
import pytorch_lightning as pl
import seaborn as sns
import torch
from hydra.utils import instantiate
from matplotlib import pyplot as plt

from bliss import generate, reporting
from bliss.catalog import FullCatalog
from bliss.datasets import sdss
from bliss.encoder import Encoder
from bliss.inference import SDSSFrame, SimulatedFrame, reconstruct_scene_at_coordinates
from bliss.models.decoder import ImageDecoder
from bliss.models.galaxy_net import OneCenteredGalaxyAE

pl.seed_everything(40)


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
        "text.latex.preamble": r"\usepackage{amsmath}",
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
        # images
        "image.cmap": "gray",
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

    if xticks is not None:
        ax.set_xticks(xticks)
    if yticks is not None:
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


class BlissFigures:
    cache = "temp.pt"

    def __init__(self, figdir, cachedir, overwrite=False, img_format="png") -> None:

        self.figdir = Path(figdir)
        self.cachefile = Path(cachedir) / self.cache
        self.overwrite = overwrite
        self.img_format = img_format

    def get_data(self, *args, **kwargs):
        """Return summary of data for producing plot, must be cachable w/ torch.save()."""
        if self.cachefile.exists() and not self.overwrite:
            return torch.load(self.cachefile)

        data = self.compute_data(*args, **kwargs)
        torch.save(data, self.cachefile)
        return data

    @abstractmethod
    def compute_data(self, *args, **kwargs):
        return {}

    def save_figures(self, *args, **kwargs):
        """Create figures and save to output directory with names from `self.fignames`."""
        data = self.get_data(*args, **kwargs)
        figs = self.create_figures(data)
        for figname, fig in figs.items():
            figfile = self.figdir / f"{figname}.{self.img_format}"
            fig.savefig(figfile, format=self.img_format)
            plt.close(fig)

    @abstractmethod
    def create_figures(self, data):
        """Return matplotlib figure instances to save based on data."""
        return {"temp_fig": mpl.figure.Figure()}


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
        true_params = image_data["params"]
        images = image_data["images"]
        recon_means = torch.tensor([])
        background = image_data["background"].reshape(1, 1, 53, 53).to(device)
        noiseless_images = image_data["noiseless"].numpy()  # no background or noise.
        snr = image_data["snr"].reshape(-1).numpy()

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
        assert np.all(recon_no_background.sum(axis=(1, 2, 3)) > 0)
        measurements = reporting.get_single_galaxy_measurements(
            slen=53,
            true_images=noiseless_images,
            recon_images=recon_no_background,
            psf_image=psf_image.reshape(-1, 53, 53),
            pixel_scale=sdss_pixel_scale,
        )
        measurements["snr"] = snr
        return {
            "random": (images[rand_indices], recon_means[rand_indices], residuals[rand_indices]),
            "worst": (images[worst_indices], recon_means[worst_indices], residuals[worst_indices]),
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

    def make_scatter_contours_plot(self, meas):
        sns.set_theme(style="darkgrid")
        set_rc_params(
            fontsize=22, legend_fontsize="small", tick_label_size="small", label_size="medium"
        )
        fig, axes = plt.subplots(1, 3, figsize=(21, 7))
        ax1, ax2, ax3 = axes.flatten()

        # fluxes / magnitudes
        x, y = meas["true_mags"], meas["recon_mags"]
        mag_ticks = (15, 16, 17, 18, 19, 20, 21, 22, 23)
        xlabel = r"$m^{\rm true}$"
        ylabel = r"$m^{\rm recon}$"
        self.make_scatter_contours(
            ax1,
            x,
            y,
            xlabel=xlabel,
            ylabel=ylabel,
            xticks=mag_ticks,
            yticks=mag_ticks,
            xlims=(15, 24),
            ylims=(15, 24),
        )
        ax1.plot([15, 24], [15, 24], color="r", lw=2)

        # ellipticities 1
        # NOTE: remove outliers for plotting purposes (contours get crazy)
        x, y = remove_outliers(meas["true_ellip"][:, 0], meas["recon_ellip"][:, 0], level=0.99)
        g_ticks = (-1.0, -0.5, 0.0, 0.5, 1.0)
        xlabel = r"$g_{1}^{\rm true}$"
        ylabel = r"$g_{1}^{\rm recon}$"
        self.make_scatter_contours(
            ax2,
            x,
            y,
            xticks=g_ticks,
            yticks=g_ticks,
            xlabel=xlabel,
            ylabel=ylabel,
            xlims=(-1.0, 1.0),
            ylims=(-1.0, 1.0),
        )
        ax2.plot([-1, 1], [-1, 1], color="r", lw=2)

        # ellipticities 2
        # NOTE: remove outliers for plotting purposes (contours get crazy)
        x, y = remove_outliers(meas["true_ellip"][:, 1], meas["recon_ellip"][:, 1], level=0.99)
        xlabel = r"$g_{2}^{\rm true}$"
        ylabel = r"$g_{2}^{\rm recon}$"
        self.make_scatter_contours(
            ax3,
            x,
            y,
            xticks=g_ticks,
            yticks=g_ticks,
            xlabel=xlabel,
            ylabel=ylabel,
            xlims=(-1.0, 1.0),
            ylims=(-1.0, 1.0),
        )
        ax3.plot([-1, 1], [-1, 1], color="r", lw=2)

        plt.tight_layout()
        return fig

    def make_scatter_bin_plots(self, meas):
        fig, axes = plt.subplots(1, 3, figsize=(21, 9))
        ax1, ax2, ax3 = axes.flatten()
        set_rc_params(fontsize=24)
        snr = meas["snr"]
        xticks = [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
        xlims = (0, 3)
        xlabel = r"$\log_{10} \text{SNR}$"

        # fluxes / magnitudes
        true_mags, recon_mags = meas["true_mags"], meas["recon_mags"]
        x, y = np.log10(snr), recon_mags - true_mags
        self.scatter_bin_plot(
            ax1,
            x,
            y,
            delta=0.2,
            xlims=xlims,
            xlabel=xlabel,
            ylabel=r"\rm $m^{\rm recon} - m^{\rm true}$",
            xticks=xticks,
        )

        # ellipticities
        true_ellip1, recon_ellip1 = meas["true_ellip"][:, 0], meas["recon_ellip"][:, 0]
        x, y = np.log10(snr), recon_ellip1 - true_ellip1
        self.scatter_bin_plot(
            ax2,
            x,
            y,
            delta=0.2,
            xlims=xlims,
            xticks=xticks,
            xlabel=xlabel,
            ylabel=r"$g_{1}^{\rm recon} - g_{1}^{\rm true}$",
        )

        true_ellip2, recon_ellip2 = meas["true_ellip"][:, 1], meas["recon_ellip"][:, 1]
        x, y = np.log10(snr), recon_ellip2 - true_ellip2
        self.scatter_bin_plot(
            ax3,
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

    def create_figures(self, data):

        return {
            "random_recon": self.reconstruction_figure(*data["random"]),
            "worst_recon": self.reconstruction_figure(*data["worst"]),
            "single_galaxy_meas_contours": self.make_scatter_contours_plot(data["measurements"]),
            "single_galaxy_meas_bins": self.make_scatter_bin_plots(data["measurements"]),
        }


class DetectionClassificationFigures(BlissFigures):
    cache = "detect_class.pt"

    @staticmethod
    def compute_mag_bin_metrics(mag_bins: np.ndarray, truth: FullCatalog, pred: FullCatalog):
        metrics_per_mag = defaultdict(lambda: np.zeros(len(mag_bins)))

        # compute data for precision/recall/classification accuracy as a function of magnitude.
        for ii, (mag1, mag2) in enumerate(mag_bins):
            res = reporting.scene_metrics(truth, pred, mag_min=mag1, mag_max=mag2, slack=1.0)
            metrics_per_mag["precision"][ii] = res["precision"].item()
            metrics_per_mag["recall"][ii] = res["recall"].item()
            metrics_per_mag["f1"][ii] = res["f1"].item()
            metrics_per_mag["class_acc"][ii] = res["class_acc"].item()
            conf_matrix = res["conf_matrix"]
            metrics_per_mag["galaxy_acc"][ii] = conf_matrix[0, 0] / conf_matrix[0, :].sum().item()
            metrics_per_mag["star_acc"][ii] = conf_matrix[1, 1] / conf_matrix[1, :].sum().item()
            for k, v in res["counts"].items():
                metrics_per_mag[k][ii] = v

        return dict(metrics_per_mag)

    def compute_metrics(self, truth: FullCatalog, pred: FullCatalog):

        # prepare magnitude bins
        mag_cuts2 = np.arange(18, 24.5, 0.25)
        mag_cuts1 = np.full_like(mag_cuts2, fill_value=-np.inf)
        mag_cuts = np.column_stack((mag_cuts1, mag_cuts2))

        mag_bins2 = np.arange(18, 25, 1.0)
        mag_bins1 = mag_bins2 - 1
        mag_bins = np.column_stack((mag_bins1, mag_bins2))

        # compute metrics
        cuts_data = self.compute_mag_bin_metrics(mag_cuts, truth, pred)
        bins_data = self.compute_mag_bin_metrics(mag_bins, truth, pred)

        # data for scatter plot of misclassifications (over all magnitudes).
        tplocs = truth.plocs.reshape(-1, 2)
        eplocs = pred.plocs.reshape(-1, 2)
        tindx, eindx, dkeep, _ = reporting.match_by_locs(tplocs, eplocs, slack=1.0)

        # compute egprob separately for PHOTO
        egbool = pred["galaxy_bools"].reshape(-1)[eindx][dkeep]
        egprob = pred.get("galaxy_probs", None)
        egprob = egbool if egprob is None else egprob.reshape(-1)[eindx][dkeep]
        full_metrics = {
            "tgbool": truth["galaxy_bools"].reshape(-1)[tindx][dkeep],
            "egbool": egbool,
            "egprob": egprob,
            "tmag": truth["mags"].reshape(-1)[tindx][dkeep],
            "emag": pred["mags"].reshape(-1)[eindx][dkeep],
        }

        return mag_cuts2, mag_bins2, cuts_data, bins_data, full_metrics

    def compute_data(
        self,
        frame: Union[SDSSFrame, SimulatedFrame],
        photo_cat: sdss.PhotoFullCatalog,
        encoder,
        decoder,
    ):
        slen = 300  # chunk side-length for whole iamge.
        bp = encoder.border_padding
        device = encoder.device
        h, w = bp, bp
        h_end = ((frame.image.shape[2] - 2 * bp) // 4) * 4 + bp
        w_end = ((frame.image.shape[3] - 2 * bp) // 4) * 4 + bp
        coadd_params: FullCatalog = frame.get_catalog((h, h_end), (w, w_end))
        photo_catalog_at_hw = photo_cat.crop_at_coords(h, h_end, w, w_end)

        # obtain predictions from BLISS.
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
        est_params = tile_est_params.cpu().to_full_params()
        est_params["fluxes"] = (
            est_params["galaxy_bools"] * est_params["galaxy_fluxes"]
            + est_params["star_bools"] * est_params["fluxes"]
        )
        est_params["mags"] = sdss.convert_flux_to_mag(est_params["fluxes"])

        # compute metrics with bliss vs coadd and photo (frame) vs coadd
        bliss_metrics = self.compute_metrics(coadd_params, est_params)
        photo_metrics = self.compute_metrics(coadd_params, photo_catalog_at_hw)

        return {"bliss_metrics": bliss_metrics, "photo_metrics": photo_metrics}

    @staticmethod
    def make_detection_figure(
        mags,
        data,
        cuts_or_bins="cuts",
        xlims=(18, 24),
        ylims=(0.5, 1.05),
        ratio=2,
        where_step="mid",
        n_gap=50,
    ):
        # precision / recall / f1 score
        assert cuts_or_bins in {"cuts", "bins"}
        precision = data["precision"]
        recall = data["recall"]
        f1_score = data["f1"]
        tgcount = data["tgcount"]
        tscount = data["tscount"]
        egcount = data["egcount"]
        escount = data["escount"]
        # (1) precision / recall
        set_rc_params(tick_label_size=22, label_size=30)
        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(10, 10), gridspec_kw={"height_ratios": [1, ratio]}, sharex=True
        )
        ymin = min(min(precision), min(recall))
        yticks = np.arange(np.round(ymin, 1), 1.1, 0.1)
        format_plot(ax2, xlabel=r"\rm magnitude cut", ylabel="metric", yticks=yticks)
        ax2.plot(mags, recall, "-o", label=r"\rm recall")
        ax2.plot(mags, precision, "-o", label=r"\rm precision")
        ax2.plot(mags, f1_score, "-o", label=r"\rm f1 score")
        ax2.legend(loc="lower left", prop={"size": 22})
        ax2.set_xlim(xlims)
        ax2.set_ylim(ylims)

        # setup histogram plot up top.
        c1 = CB_color_cycle[3]
        c2 = CB_color_cycle[4]
        ax1.step(mags, tgcount, label="coadd galaxies", where=where_step, color=c1)
        ax1.step(mags, tscount, label="coadd stars", where=where_step, color=c2)
        ax1.step(mags, egcount, label="pred. galaxies", ls="--", where=where_step, color=c1)
        ax1.step(mags, escount, label="pred. stars", ls="--", where=where_step, color=c2)
        ymax = max(max(tgcount), max(tscount), max(egcount), max(escount))
        ymax = np.ceil(ymax / n_gap) * n_gap
        yticks = np.arange(0, ymax, n_gap)
        ax1.set_ylim((0, ymax))
        format_plot(ax1, yticks=yticks, ylabel=r"\rm Counts")
        ax1.legend(loc="best", prop={"size": 16})
        plt.subplots_adjust(hspace=0)

        return fig

    @staticmethod
    def make_classification_figure(
        mags,
        data,
        cuts_or_bins="cuts",
        xlims=(18, 24),
        ylims=(0.5, 1.05),
        ratio=2,
        where_step="mid",
        n_gap=50,
    ):
        # classification accuracy
        class_acc = data["class_acc"]
        galaxy_acc = data["galaxy_acc"]
        star_acc = data["star_acc"]
        set_rc_params(tick_label_size=22, label_size=30)
        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(10, 10), gridspec_kw={"height_ratios": [1, ratio]}, sharex=True
        )
        xlabel = r"\rm magnitude " + cuts_or_bins[:-1]
        format_plot(ax2, xlabel=xlabel, ylabel="classification accuracy")
        ax2.plot(mags, galaxy_acc, "-o", label=r"\rm galaxy")
        ax2.plot(mags, star_acc, "-o", label=r"\rm star")
        ax2.plot(mags, class_acc, "-o", label=r"\rm overall")
        ax2.set_xlim(xlims)
        ax2.set_ylim(ylims)
        ax2.legend(loc="lower left", prop={"size": 18})

        # setup histogram up top.
        gcounts = data["n_matches_coadd_gal"]
        scounts = data["n_matches_coadd_star"]
        ax1.step(mags, gcounts, label=r"\rm matched coadd galaxies", where=where_step)
        ax1.step(mags, scounts, label=r"\rm matched coadd stars", where=where_step)
        ymax = max(max(gcounts), max(scounts))
        ymax = np.ceil(ymax / n_gap) * n_gap
        yticks = np.arange(0, ymax, n_gap)
        format_plot(ax1, yticks=yticks, ylabel=r"\rm Counts")
        ax1.legend(loc="best", prop={"size": 16})
        ax1.set_ylim((0, ymax))
        plt.subplots_adjust(hspace=0)

        return fig

    @staticmethod
    def make_magnitude_prob_scatter_figure(data):
        # scatter of matched objects magnitude vs classification probability.
        set_rc_params(tick_label_size=22, label_size=30)
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        tgbool = data["tgbool"].numpy().astype(bool)
        egbool = data["egbool"].numpy().astype(bool)
        tmag, egprob = data["tmag"].numpy(), data["egprob"].numpy()
        correct = np.equal(tgbool, egbool)

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

        return fig

    @staticmethod
    def make_mag_mag_scatter_figure(data):
        tgbool = data["tgbool"].numpy().astype(bool)
        tmag, emag = data["tmag"].numpy(), data["emag"].numpy()

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 7))
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

        return fig

    def create_metrics_figures(
        self, mag_cuts, mag_bins, cuts_data, bins_data, full_metrics, name=""
    ):
        f1 = self.make_detection_figure(mag_cuts, cuts_data, "cuts", ylims=(0.5, 1.03))
        f2 = self.make_classification_figure(mag_cuts, cuts_data, "cuts", ylims=(0.8, 1.03))
        f3 = self.make_detection_figure(
            mag_bins - 0.5, bins_data, "bins", xlims=(17, 24), ylims=(0.0, 1.05), n_gap=25
        )
        f4 = self.make_classification_figure(
            mag_bins - 0.5, bins_data, "bins", xlims=(17, 24), ylims=(0.0, 1.05), n_gap=25
        )
        f5 = self.make_magnitude_prob_scatter_figure(full_metrics)
        f6 = self.make_mag_mag_scatter_figure(full_metrics)

        return {
            f"{name}_detection_cuts": f1,
            f"{name}_class_cuts": f2,
            f"{name}_detection_bins": f3,
            f"{name}_class_bins": f4,
            f"{name}_mag_prob_scatter": f5,
            f"{name}_mag_mag_scatter": f6,
        }

    def create_figures(self, data):
        """Make figures related to detection and classification in SDSS."""
        sns.set_theme(style="darkgrid")
        bliss_figs = self.create_metrics_figures(*data["bliss_metrics"], name="bliss_sdss")
        photo_figs = self.create_metrics_figures(*data["photo_metrics"], name="photo_sdss")
        return {**bliss_figs, **photo_figs}


class SDSSReconstructionFigures(BlissFigures):
    cache = "recon_sdss.pt"

    def __init__(self, scenes, figdir, cachedir, overwrite=False, img_format="png") -> None:
        self.scenes = scenes
        super().__init__(figdir, cachedir, overwrite=overwrite, img_format=img_format)

    def compute_data(self, frame: Union[SDSSFrame, SimulatedFrame], encoder: Encoder, decoder):

        tile_slen = encoder.location_encoder.tile_slen
        device = encoder.device
        data = {}

        for figname, scene_coords in self.scenes.items():
            h, w, scene_size = scene_coords["h"], scene_coords["w"], scene_coords["size"]
            assert h % tile_slen == 0 and w % tile_slen == 0
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

            tile_map_recon = tile_map_recon.cpu()
            recon_map = tile_map_recon.to_full_params()

            # get BLISS probability of n_sources in coadd locations.
            coplocs = coadd_params.plocs.reshape(-1, 2)
            prob_n_sources = tile_map_recon.get_tile_params_at_coord(coplocs)["n_source_log_probs"]
            prob_n_sources = prob_n_sources.exp()

            true = true.cpu()
            recon = recon.cpu()
            resid = resid.cpu()
            data[figname] = (true, recon, resid, coadd_params, recon_map, prob_n_sources)

        return data

    def create_figures(self, data):  # pylint: disable=too-many-statements
        """Make figures related to reconstruction in SDSS."""
        out_figures = {}

        pad = 6.0
        set_rc_params(fontsize=22, tick_label_size="small", legend_fontsize="small")
        for figname, scene_coords in self.scenes.items():
            scene_size = scene_coords["size"]
            true, recon, res, coadd_params, recon_map, prob_n_sources = data[figname]
            fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(28, 12))

            ax_true = axes[0]
            ax_recon = axes[1]
            ax_res = axes[2]

            ax_true.set_title("Original Image", pad=pad)
            ax_recon.set_title("Reconstruction", pad=pad)
            ax_res.set_title("Residual", pad=pad)

            s = 55 * 300 / scene_size  # marker size
            lw = 2 * np.sqrt(300 / scene_size)

            vrange1 = (800, 1100)
            vrange2 = (-5, 5)
            labels = ["Coadd Galaxies", "Coadd Stars", "BLISS Galaxies", "BLISS Stars"]
            reporting.plot_image_and_locs(
                fig, ax_true, 0, true, 0, coadd_params, recon_map, vrange1, s, lw
            )
            reporting.plot_image_and_locs(
                fig, ax_recon, 0, recon, 0, coadd_params, recon_map, vrange1, s, lw, labels=labels
            )
            reporting.plot_image_and_locs(
                fig, ax_res, 0, res, 0, coadd_params, recon_map, vrange2, s, lw, 0.5
            )
            plt.subplots_adjust(hspace=-0.4)
            plt.tight_layout()

            # plot probability of detection in each true object for blends
            if "blend" in figname:
                for ii, ploc in enumerate(coadd_params.plocs.reshape(-1, 2)):
                    prob = prob_n_sources[ii].item()
                    x, y = ploc[1] + 0.5, ploc[0] + 0.5
                    text = r"$\boldsymbol{" + f"{prob:.2f}" + "}$"
                    ax_true.annotate(text, (x, y), color="lime")

            out_figures[figname] = fig

        return out_figures


def load_models(cfg, device):
    # load models required for SDSS reconstructions.

    location = instantiate(cfg.models.location_encoder).to(device).eval()
    location.load_state_dict(
        torch.load(cfg.predict.location_checkpoint, map_location=location.device)
    )

    binary = instantiate(cfg.models.binary).to(device).eval()
    binary.load_state_dict(torch.load(cfg.predict.binary_checkpoint, map_location=binary.device))

    galaxy = instantiate(cfg.models.galaxy_encoder).to(device).eval()
    galaxy.load_state_dict(torch.load(cfg.predict.galaxy_checkpoint, map_location=galaxy.device))

    encoder = Encoder(location.eval(), binary.eval(), galaxy.eval())
    encoder = encoder.to(device)

    decoder: ImageDecoder = instantiate(cfg.models.decoder).to(device).eval()

    return encoder, decoder


@hydra.main(config_path="./config", config_name="config")
def main(cfg):  # pylint: disable=too-many-statements

    figs = set(cfg.plots.figs)
    cachedir = cfg.plots.cachedir
    device = torch.device(cfg.plots.device)
    bfig_kwargs = {
        "figdir": cfg.plots.figdir,
        "cachedir": cachedir,
        "overwrite": cfg.plots.overwrite,
        "img_format": cfg.plots.image_format,
    }

    if not Path(cachedir).exists():
        warnings.warn("Specified cache directory does not exist, will attempt to create it.")
        Path(cachedir).mkdir(exist_ok=True, parents=True)

    if figs.intersection({2, 3}):
        # load SDSS frame and models for prediction
        frame: Union[SDSSFrame, SimulatedFrame] = instantiate(cfg.plots.frame)
        photo_cat = sdss.PhotoFullCatalog.from_file(**cfg.plots.photo_catalog)

        encoder, decoder = load_models(cfg, device)

    # FIGURE 1: Autoencoder single galaxy reconstruction
    if 1 in figs:
        print("INFO: Creating autoencoder figures...")
        autoencoder = instantiate(cfg.models.galaxy_net)
        autoencoder.load_state_dict(torch.load(cfg.models.prior.galaxy_prior.autoencoder_ckpt))
        autoencoder = autoencoder.to(device).eval()

        # generate galsim simulated galaxies images if file does not exist.
        galaxies_file = Path(cfg.plots.simulated_sdss_individual_galaxies)
        if not galaxies_file.exists() or cfg.plots.overwrite:
            print(f"INFO: Generating individual galaxy images and saving to: {galaxies_file}")
            dataset = instantiate(
                cfg.datasets.sdss_galaxies, batch_size=512, n_batches=20, num_workers=20
            )
            imagepath = galaxies_file.parent / (galaxies_file.stem + "_images.png")
            generate.generate(
                dataset, galaxies_file, imagepath, n_plots=25, global_params=("background", "slen")
            )

        # create figure classes and plot.
        ae_figures = AEReconstructionFigures(n_examples=5, **bfig_kwargs)
        ae_figures.save_figures(
            autoencoder, galaxies_file, cfg.plots.psf_file, cfg.plots.sdss_pixel_scale
        )
        mpl.rc_file_defaults()

    # FIGURE 2: Classification and Detection metrics
    if 2 in figs:
        print("INFO: Creating classification and detection metrics from SDSS frame figures...")
        dc_fig = DetectionClassificationFigures(**bfig_kwargs)
        dc_fig.save_figures(frame, photo_cat, encoder, decoder)
        mpl.rc_file_defaults()

    # FIGURE 3: Reconstructions on SDSS
    if 3 in figs:
        print("INFO: Creating reconstructions from SDSS figures...")
        sdss_rec_fig = SDSSReconstructionFigures(cfg.plots.scenes, **bfig_kwargs)
        sdss_rec_fig.save_figures(frame, encoder, decoder)
        mpl.rc_file_defaults()

    if not figs.intersection({1, 2, 3}):
        raise NotImplementedError(
            "No figures were created, `cfg.plots.figs` should be a subset of [1,2,3]."
        )


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
