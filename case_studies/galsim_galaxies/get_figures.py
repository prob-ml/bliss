#!/usr/bin/env python3
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import hydra
import matplotlib as mpl
import numpy as np
import pytorch_lightning as pl
import torch
from hydra.utils import instantiate
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from torch import Tensor
from tqdm import tqdm

from bliss import generate, metrics
from bliss.catalog import FullCatalog, TileCatalog
from bliss.datasets.galsim_galaxies import GalsimBlends
from bliss.datasets.sdss import convert_mag_to_flux
from bliss.encoder import Encoder
from bliss.metrics import compute_bin_metrics, get_boostrap_precision_and_recall, match_by_locs
from bliss.models.decoder import ImageDecoder
from bliss.models.galaxy_net import OneCenteredGalaxyAE
from bliss.models.psf_decoder import PSFDecoder
from bliss.plotting import BlissFigure, plot_image, scatter_shade_plot

ALL_FIGS = ("single_gal", "blend_gal", "toy")

pl.seed_everything(42)


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
            ax_true.set_title("Images $x$", pad=pad)
            ax_recon.set_title(r"Reconstruction $\tilde{x}$", pad=pad)
            ax_res.set_title(r"Residual $\left(x - \tilde{x}\right) / \sqrt{\tilde{x}}$", pad=pad)

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


def _make_pr_figure(
    bins: np.ndarray,
    data: Dict[str, np.ndarray],
    xlabel: str,
    xlims: Tuple[float, float] = None,
    ylims: Tuple[float, float] = None,
    ylims2: Tuple[float, float] = None,
    ratio: float = 2,
    where_step: str = "mid",
    n_ticks: int = 5,
    ordmag: int = 3,
    metric_type: str = "Detection",
    legend_size_hist: int = 20,
):
    precision = data["precision"]
    recall = data["recall"]
    boot_precision = data["boot"]["precision"]
    boot_recall = data["boot"]["recall"]
    tgcount = data["tgcount"]
    tscount = data["tscount"]
    egcount = data["egcount"]
    escount = data["escount"]

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(10, 10), gridspec_kw={"height_ratios": [1, ratio]}, sharex=True
    )

    # (bottom) plot of precision and recall
    ymin = min(precision.min(), recall.min())
    yticks = np.arange(np.round(ymin, 1), 1.1, 0.1)
    c1 = plt.rcParams["axes.prop_cycle"].by_key()["color"][0]
    precision1 = np.quantile(boot_precision, 0.25, 0)
    precision2 = np.quantile(boot_precision, 0.75, 0)
    ax2.plot(bins, precision, "-o", color=c1, label=r"\rm Precision", markersize=6)
    ax2.fill_between(bins, precision1, precision2, color=c1, alpha=0.5)

    c2 = plt.rcParams["axes.prop_cycle"].by_key()["color"][1]
    recall1 = np.quantile(boot_recall, 0.25, 0)
    recall2 = np.quantile(boot_recall, 0.75, 0)
    ax2.plot(bins, recall, "-o", color=c2, label=r"\rm Recall", markersize=6)
    ax2.fill_between(bins, recall1, recall2, color=c2, alpha=0.5)

    ax2.legend(loc="lower left")
    ax2.set_xlabel(xlabel)
    ax2.set_ylabel(rf"\rm {metric_type} Metric")
    ax2.set_yticks(yticks)

    if xlims is not None:
        ax2.set_xlim(xlims)
    if ylims is not None:
        ax2.set_ylim(ylims)
    if ylims2 is not None:
        ax1.set_ylim(ylims2)

    # setup histogram plot up top
    c1 = plt.rcParams["axes.prop_cycle"].by_key()["color"][3]
    c2 = plt.rcParams["axes.prop_cycle"].by_key()["color"][4]
    ax1.step(bins, tgcount, label="True galaxies", where=where_step, color=c1)
    ax1.step(bins, tscount, label="True stars", where=where_step, color=c2)
    ax1.step(bins, egcount, label="Pred. galaxies", ls="--", where=where_step, color=c1)
    ax1.step(bins, escount, label="Pred. stars", ls="--", where=where_step, color=c2)
    ymax = max(max(tgcount), max(tscount), max(egcount), max(escount))
    yticks = np.round(np.linspace(0, ymax, n_ticks), -ordmag)
    ax1.set_yticks(yticks)
    ax1.set_ylabel(r"\rm Counts")
    ax1.legend(loc="best", prop={"size": legend_size_hist})
    plt.subplots_adjust(hspace=0)
    return fig


class AutoEncoderReconRandom(BlissFigure):
    def __init__(self, *args, n_examples: int = 5, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.n_examples = n_examples

    @property
    def rc_kwargs(self):
        return {"fontsize": 22, "tick_label_size": "small", "legend_fontsize": "small"}

    @property
    def cache_name(self) -> str:
        return "ae"

    @property
    def name(self) -> str:
        return "ae_recon_random"

    def compute_data(
        self,
        autoencoder: OneCenteredGalaxyAE,
        images_file: str,
        psf_params_file: str,
        sdss_pixel_scale: float,
    ):
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
        snr_thres = 10
        high_snr_indices = torch.where(snr > snr_thres)[0]
        rand_perm_indx = torch.randperm(len(high_snr_indices))[: self.n_examples]
        rand_indices = high_snr_indices[rand_perm_indx]

        # worst
        q = 100
        absolute_residual = residuals.abs().sum(axis=(1, 2, 3))
        worst_indices = absolute_residual.argsort()[-self.n_examples - q : -q]

        # measurements
        psf_decoder = PSFDecoder(psf_params_file=psf_params_file, psf_slen=53, sdss_bands=[2])
        psf_image = psf_decoder.forward_psf_from_params()

        recon_no_background = recon_means - background.cpu()
        assert torch.all(recon_no_background.sum(axis=(1, 2, 3)) > 0)
        true_meas = metrics.get_single_galaxy_measurements(
            noiseless_images, psf_image.reshape(-1, 53, 53), sdss_pixel_scale
        )
        true_meas = {f"true_{k}": v for k, v in true_meas.items()}
        recon_meas = metrics.get_single_galaxy_measurements(
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

    def _reconstruction_figure(self, images, recons, residuals) -> Figure:
        pad = 6.0
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
            plot_image(fig, ax_true, image, vrange=(vmin, vmax))
            plot_image(fig, ax_recon, recon, vrange=(vmin, vmax))
            plot_image(fig, ax_res, residual, vrange=(vmin_res, vmax_res))

        plt.subplots_adjust(hspace=-0.4)
        plt.tight_layout()

        return fig

    def create_figure(self, data) -> Figure:
        return _reconstruction_figure(self.n_examples, *data["random"].values())


class AutoEncoderBinMeasurements(AutoEncoderReconRandom):
    @property
    def name(self) -> str:
        return "ae_bin_measurements"

    @property
    def rc_kwargs(self):
        return {"fontsize": 24}

    def create_figure(self, data) -> Figure:
        meas = data["measurements"]

        fig, axes = plt.subplots(1, 3, figsize=(18, 7))
        ax1, ax2, ax3 = axes.flatten()
        snr = meas["snr"]
        xticks = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
        xlims = (0.5, 3)
        xlabel = r"$\log_{10} \text{SNR}$"

        # fluxes
        true_fluxes, recon_fluxes = meas["true_fluxes"], meas["recon_fluxes"]
        x, y = np.log10(snr), (true_fluxes - recon_fluxes) / true_fluxes
        scatter_shade_plot(ax1, x, y, xlims, delta=0.2)
        ax1.set_xlim(xlims)
        ax1.set_xlabel(xlabel)
        ax1.set_ylabel(r"\rm $(f^{\rm true} - f^{\rm recon}) / f^{\rm true}$")
        ax1.set_xticks(xticks)
        ax1.axhline(0, ls="--", color="k")

        # ellipticities
        true_ellip1, recon_ellip1 = meas["true_ellips"][:, 0], meas["recon_ellips"][:, 0]
        x, y = np.log10(snr), true_ellip1 - recon_ellip1
        scatter_shade_plot(ax2, x, y, xlims, delta=0.2)
        ax2.set_xlim(xlims)
        ax2.set_xlabel(xlabel)
        ax2.set_ylabel(r"$g_{1}^{\rm true} - g_{1}^{\rm recon}$")
        ax2.set_xticks(xticks)
        ax2.axhline(0, ls="--", color="k")

        true_ellip2, recon_ellip2 = meas["true_ellips"][:, 1], meas["recon_ellips"][:, 1]
        x, y = np.log10(snr), true_ellip2 - recon_ellip2
        scatter_shade_plot(ax3, x, y, xlims, delta=0.2)
        ax3.set_xlim(xlims)
        ax3.set_xlabel(xlabel)
        ax3.set_ylabel(r"$g_{2}^{\rm true} - g_{2}^{\rm recon}$")
        ax3.set_xticks(xticks)
        ax3.axhline(0, ls="--", color="k")

        return fig


class AutoEncoderReconWorst(AutoEncoderReconRandom):
    @property
    def name(self) -> str:
        return "ae_recon_worst"

    @property
    def rc_kwargs(self):
        return {"fontsize": 24}

    def create_figure(self, data) -> Figure:
        return _reconstruction_figure(self.n_examples, *data["worst"].values())


class AutoEncoderHistograms(AutoEncoderReconRandom):
    @property
    def name(self) -> str:
        return "ae_bin_hists"

    @property
    def rc_kwargs(self):
        return {"fontsize": 24}

    def create_figure(self, data) -> Figure:
        snr = data["measurements"]["snr"]
        fig, ax = plt.subplots(1, 1, figsize=(7, 7))
        xticks = [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
        xlims = (0, 3)
        snr_bins = np.arange(xlims[0], xlims[1] + 0.2, 0.2)
        xlabel = r"$\log_{10} \text{SNR}$"
        ax.hist(np.log10(snr), bins=snr_bins, histtype="step")
        ax.set_xlabel(xlabel)
        ax.set_xticks(xticks)
        ax.axvline(np.log10(snr).mean(), label=r"\rm Mean", color="k", ls="--")

        ax.legend(prop={"size": 18})

        return fig


class BlendResidualFigure(BlissFigure):
    @property
    def rc_kwargs(self):
        return {"fontsize": 24}

    @property
    def cache_name(self) -> str:
        return "blend_sim"

    @property
    def name(self) -> str:
        return "blendsim_gal_meas"

    def compute_data(self, blend_file: Path, encoder: Encoder, decoder: ImageDecoder):
        blend_data: Dict[str, Tensor] = torch.load(blend_file)
        images = blend_data.pop("images")
        background = blend_data.pop("background")
        n_batches, _, slen, _ = images.shape
        assert background.shape == (1, slen, slen)

        # prepare background
        background = background.unsqueeze(0)
        background = background.expand(n_batches, 1, slen, slen)

        # background for snr
        bg = background[0, 0, 0, 0].item()

        # first create FullCatalog from simulated data
        tile_cat = TileCatalog(decoder.tile_slen, blend_data).cpu()
        truth = tile_cat.to_full_params()

        print("INFO: BLISS posterior inference on images.")
        tile_est = encoder.variational_mode(images, background)
        tile_est.set_all_fluxes_and_mags(decoder)
        tile_est.set_galaxy_ellips(decoder, scale=0.393)
        tile_est.set_snr(decoder, bg)
        tile_est = tile_est.cpu()
        est = tile_est.to_full_params()

        # compute detection metrics (snr)
        snr_bins2 = 10 ** torch.arange(0.2, 3.2, 0.2)
        snr_bins1 = 10 ** torch.arange(0.0, 3.0, 0.2)
        snr_bins = torch.column_stack((snr_bins1, snr_bins2))
        bin_metrics = compute_bin_metrics(truth, est, "snr", snr_bins)
        boot_metrics = get_boostrap_precision_and_recall(1000, truth, est, "snr", snr_bins)

        # collect quantities for residuals on ellipticites and flux of galaxies
        snr = []
        blendedness = []
        true_mags = []
        true_ellips1 = []
        true_ellips2 = []
        est_mags = []
        est_ellips1 = []
        est_ellips2 = []
        snr_class = []
        mag_class = []
        tgbools = []
        egbools = []
        for ii in tqdm(range(n_batches), desc="Matching batches"):
            true_plocs_ii, est_plocs_ii = truth.plocs[ii], est.plocs[ii]
            tindx, eindx, dkeep, _ = match_by_locs(true_plocs_ii, est_plocs_ii)
            n_matches = len(tindx[dkeep])

            if n_matches > 0:
                # only evaluate flux/ellipticity residuals on galaxies labelled as galaxies.
                tgbool_ii = truth["galaxy_bools"][ii][tindx][dkeep]
                egbool_ii = est["galaxy_bools"][ii][eindx][dkeep]
                gbool_ii = torch.eq(tgbool_ii, egbool_ii).eq(torch.ones_like(tgbool_ii))
                gbool_ii = gbool_ii.flatten()
                snr_ii_class = truth["snr"][ii][tindx][dkeep]
                mag_ii_class = truth["mags"][ii][tindx][dkeep]

                assert len(tgbool_ii) == len(egbool_ii) == len(snr_ii_class) == n_matches

                # save snr, mag, and booleans over matches for classification metrics
                for jj in range(n_matches):
                    snr_class.append(snr_ii_class[jj].item())
                    tgbools.append(tgbool_ii[jj].item())
                    egbools.append(egbool_ii[jj].item())
                    mag_class.append(mag_ii_class[jj].item())

                snr_ii = truth["snr"][ii][tindx][dkeep][gbool_ii]  # noqa: WPS219
                blendedness_ii = truth["blendedness"][ii][tindx][dkeep][gbool_ii]  # noqa: WPS219
                true_mag_ii = truth["mags"][ii][tindx][dkeep][gbool_ii]  # noqa: WPS219
                est_mag_ii = est["mags"][ii][eindx][dkeep][gbool_ii]  # noqa: WPS219
                true_ellips_ii = truth["ellips"][ii][tindx][dkeep][gbool_ii]  # noqa: WPS219
                est_ellips_ii = est["ellips"][ii][eindx][dkeep][gbool_ii]  # noqa: WPS219

                n_matched_gals = len(snr_ii)

                for jj in range(n_matched_gals):
                    snr.append(snr_ii[jj].item())
                    blendedness.append(blendedness_ii[jj].item())
                    true_mags.append(true_mag_ii[jj].item())
                    est_mags.append(est_mag_ii[jj].item())
                    true_ellips1.append(true_ellips_ii[jj][0].item())
                    true_ellips2.append(true_ellips_ii[jj][1].item())
                    est_ellips1.append(est_ellips_ii[jj][0].item())
                    est_ellips2.append(est_ellips_ii[jj][1].item())

        true_ellips = torch.vstack([torch.tensor(true_ellips1), torch.tensor(true_ellips2)])
        true_ellips = true_ellips.T.reshape(-1, 2)

        est_ellips = torch.vstack([torch.tensor(est_ellips1), torch.tensor(est_ellips2)])
        est_ellips = est_ellips.T.reshape(-1, 2)

        true_fluxes = convert_mag_to_flux(torch.tensor(true_mags))
        est_fluxes = convert_mag_to_flux(torch.tensor(est_mags))

        return {
            "residuals": {
                "snr": torch.tensor(snr),
                "blendedness": torch.tensor(blendedness),
                "true_fluxes": true_fluxes,
                "est_fluxes": est_fluxes,
                "true_ellips": true_ellips,
                "est_ellips": est_ellips,
            },
            "detection": {
                "precision": bin_metrics["precision"],
                "recall": bin_metrics["recall"],
                "tgcount": bin_metrics["tgcount"],
                "tscount": bin_metrics["tscount"],
                "egcount": bin_metrics["egcount"],
                "escount": bin_metrics["escount"],
                "boot": {
                    "precision": boot_metrics["precision"],
                    "recall": boot_metrics["recall"],
                },
                "bins": snr_bins,
            },
            "classification": {
                "snr": torch.tensor(snr_class),
                "mags": torch.tensor(mag_class),
                "tgbools": torch.tensor(tgbools),
                "egbools": torch.tensor(egbools),
            },
        }

    def create_figure(self, data) -> Figure:
        snr, blendedness, tfluxes, efluxes, true_ellips, est_ellips = data["residuals"].values()
        fig, axes = plt.subplots(3, 2, figsize=(12, 18))
        ax1, ax2, ax3, ax4, ax5, ax6 = axes.flatten()

        xticks = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
        xlims = (0.5, 3)
        xlabel = r"$\log_{10} \rm SNR$"
        ylabel = r"\rm $(f^{\rm true} - f^{\rm recon}) / f^{\rm true}$"

        x, y = np.log10(snr), (tfluxes - efluxes) / tfluxes
        scatter_shade_plot(ax1, x, y, xlims, delta=0.2)
        ax1.set_xlabel(xlabel)
        ax1.set_ylabel(ylabel)
        ax1.set_xticks(xticks)
        ax1.axhline(0, ls="--", color="k")

        xticks = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
        xlims = (0, 1)
        xlabel = "$B$"
        ylabel = r"\rm $(f^{\rm true} - f^{\rm recon}) / f^{\rm true}$"
        x, y = blendedness, (tfluxes - efluxes) / tfluxes
        scatter_shade_plot(ax2, x, y, xlims, delta=0.1)
        ax2.set_xlabel(xlabel)
        ax2.set_ylabel(ylabel)
        ax2.set_xticks(xticks)
        ax2.axhline(0, ls="--", color="k")

        xticks = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
        xlims = (0.5, 3)
        xlabel = r"$\log_{10} \rm SNR$"
        ylabel = r"$g_{1}^{\rm true} - g_{1}^{\rm recon}$"
        x, y = np.log10(snr), true_ellips[:, 0] - est_ellips[:, 0]
        scatter_shade_plot(ax3, x, y, xlims, delta=0.2)
        ax3.set_xlabel(xlabel)
        ax3.set_ylabel(ylabel)
        ax3.set_xticks(xticks)
        ax3.axhline(0, ls="--", color="k")

        xticks = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
        xlims = (0, 1)
        xlabel = "$B$"
        ylabel = r"$g_{1}^{\rm true} - g_{1}^{\rm recon}$"
        x, y = blendedness, true_ellips[:, 0] - est_ellips[:, 0]
        scatter_shade_plot(ax4, x, y, xlims, delta=0.1)
        ax4.set_xlabel(xlabel)
        ax4.set_ylabel(ylabel)
        ax4.set_xticks(xticks)
        ax4.axhline(0, ls="--", color="k")

        xticks = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
        xlims = (0.5, 3)
        xlabel = r"$\log_{10} \rm SNR$"
        ylabel = r"$g_{2}^{\rm true} - g_{2}^{\rm recon}$"
        x, y = np.log10(snr), true_ellips[:, 1] - est_ellips[:, 1]
        scatter_shade_plot(ax5, x, y, xlims, delta=0.2)
        ax5.set_xlabel(xlabel)
        ax5.set_ylabel(ylabel)
        ax5.set_xticks(xticks)
        ax5.axhline(0, ls="--", color="k")

        xticks = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
        xlims = (0, 1)
        xlabel = "$B$"
        ylabel = r"$g_{2}^{\rm true} - g_{2}^{\rm recon}$"
        x, y = blendedness, true_ellips[:, 1] - est_ellips[:, 1]
        scatter_shade_plot(ax6, x, y, xlims=xlims, delta=0.1)
        ax6.set_xlabel(xlabel)
        ax6.set_ylabel(ylabel)
        ax6.set_xticks(xticks)
        ax6.axhline(0, ls="--", color="k")

        plt.tight_layout()

        return fig


class BlendDetectionFigure(BlendResidualFigure):
    @property
    def rc_kwargs(self):
        return {"fontsize": 32}

    @property
    def name(self) -> str:
        return "blendsim_detection"

    def create_figure(self, data) -> Figure:
        # take middle of bin as x for plotting
        snr_bins = np.log10(data["detection"]["bins"].mean(1))
        return _make_pr_figure(
            snr_bins, data["detection"], r"$\log_{10} \rm SNR$", xlims=(0.5, 3), ylims2=(0, 2000)
        )


class BlendClassificationFigure(BlendResidualFigure):
    @property
    def rc_kwargs(self):
        return {"fontsize": 28}

    @property
    def name(self) -> str:
        return "blendsim_classification"

    def _compute_pr(self, tgbool: np.ndarray, egbool: np.ndarray):
        t = np.sum(tgbool)
        p = np.sum(egbool)

        cond1 = np.equal(tgbool, egbool).astype(bool)
        cond2 = tgbool.astype(bool)
        tp = (cond1 & cond2).astype(float).sum()

        assert np.all(np.greater_equal(t, tp))
        assert np.all(np.greater_equal(p, tp))
        if t == 0 or p == 0:
            return np.nan, np.nan

        return tp / p, tp / t

    def create_figure(self, data) -> Figure:
        snrs, _, tgbools, egbools = data["classification"].values()
        snr_bins = data["detection"]["bins"]
        n_matches = len(snrs)
        n_bins = len(snr_bins)
        n_boots = 1000

        precision = np.zeros(n_bins)
        recall = np.zeros(n_bins)
        tgals = np.zeros(n_bins)
        egals = np.zeros(n_bins)
        tstars = np.zeros(n_bins)
        estars = np.zeros(n_bins)

        boot_precision = np.zeros((n_boots, n_bins))
        boot_recall = np.zeros((n_boots, n_bins))

        boot_indices = np.random.randint(0, n_matches, (n_boots, n_matches))

        # compute boostrap precision and recall per bin
        for ii in range(n_boots):
            snrs_ii = snrs[boot_indices[ii]]
            tgbools_ii = tgbools[boot_indices[ii]]
            egbools_ii = egbools[boot_indices[ii]]
            for jj, (b1, b2) in enumerate(snr_bins):
                keep = (b1 < snrs_ii) & (snrs_ii < b2)
                tgbool_ii = tgbools_ii[keep]
                egbool_ii = egbools_ii[keep]

                p, r = self._compute_pr(tgbool_ii, egbool_ii)
                boot_precision[ii][jj] = p
                boot_recall[ii][jj] = r

        # compute precision and recall per bin
        for jj, (b1, b2) in enumerate(snr_bins):
            keep = (b1 < snrs) & (snrs < b2)
            tgbool = tgbools[keep]
            egbool = egbools[keep]
            p, r = self._compute_pr(tgbool, egbool)
            precision[jj] = p
            recall[jj] = r

            tgals[jj] = tgbool.sum()
            egals[jj] = egbool.sum()
            tstars[jj] = (~tgbool.astype(bool)).astype(float).sum()
            estars[jj] = (~egbool.astype(bool)).astype(float).sum()

        bins = np.log10(snr_bins.mean(1))
        data = {
            "precision": precision,
            "recall": recall,
            "tgcount": tgals,
            "egcount": egals,
            "tscount": tstars,
            "escount": estars,
            "boot": {"precision": boot_precision, "recall": boot_recall},
        }
        return _make_pr_figure(
            bins,
            data,
            r"$\log_{10} \rm SNR$",
            xlims=(0.5, 3),
            metric_type="Galaxy Classification",
            ylims2=(0, 1000),
            legend_size_hist=16,
            ylims=(0.5, 1.03),
        )


class BlendHistogramFigure(BlendResidualFigure):
    @property
    def rc_kwargs(self):
        return {"fontsize": 32}

    @property
    def name(self) -> str:
        return "blendsim_hists"

    def create_figure(self, data) -> Figure:
        snr = np.log10(data["residuals"]["snr"])
        blendedness = data["residuals"]["blendedness"]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

        xticks = [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
        snr_bins = np.arange(0, 3.2, 0.2)
        ax1.hist(snr, bins=snr_bins, histtype="step", log=True)
        ax1.set_xlabel(r"$\log_{10} \rm SNR$")
        ax1.set_ylabel(r"\rm Number of galaxies", size=24)
        ax1.set_xticks(xticks)

        xticks = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
        blendedness_bins = np.arange(0, 1.1, 0.1)
        ax2.hist(blendedness, bins=blendedness_bins, histtype="step", log=True)
        ax2.set_xlabel("$B$")
        ax2.set_ylabel(r"\rm Number of galaxies", size=24)
        ax2.set_xticks(xticks)

        return fig


class ToySeparationFigure(BlissFigure):
    @property
    def rc_kwargs(self):
        return {"fontsize": 22, "tick_label_size": "small", "legend_fontsize": "small"}

    @property
    def cache_name(self) -> str:
        return "toy_separation"

    @property
    def name(self) -> str:
        return "toy_separations"

    @property
    def separations_to_plot(self) -> List[int]:
        return [4, 8, 12]

    def compute_data(self, encoder: Encoder, decoder: ImageDecoder, blends_ds: GalsimBlends):
        # first, decide image size
        slen = 44
        bp = encoder.detection_encoder.border_padding
        tile_slen = encoder.detection_encoder.tile_slen
        size = 44 + 2 * bp
        blends_ds.slen = slen
        blends_ds.decoder.slen = slen
        assert slen / tile_slen % 2 == 1, "Need odd number of tiles to center galaxy."

        # now separations between galaxies to be considered (in pixels)
        # for efficiency, we set the batch_size equal to the number of separations
        seps = torch.arange(0, 18, 0.1)
        batch_size = len(seps)

        # Params: total_flux, disk_frac, beta_radians, disk_q, disk_a, bulge_q, bulge_a
        # first centered galaxy, then moving one.
        n_sources = 2
        flux1, flux2 = 2e5, 1e5
        gparams = torch.tensor(
            [
                [flux1, 1.0, torch.pi / 4, 0.7, 1.5, 0, 0],
                [flux2, 1.0, 3 * torch.pi / 4, 0.7, 1.0, 0, 0],
            ],
        )
        gparams = gparams.reshape(1, 2, 7).expand(batch_size, 2, 7)

        # create full catalogs (need separately since decoder only accepts 1 batch)
        x0, y0 = 22, 22  # center plocs
        images = torch.zeros(batch_size, 1, size, size)
        background = torch.zeros(batch_size, 1, size, size)
        plocs = torch.tensor([[[x0, y0], [x0, y0 + sep]] for sep in seps]).reshape(batch_size, 2, 2)
        for ii in range(batch_size):
            ploc = plocs[ii].reshape(1, 2, 2)
            d = {
                "n_sources": torch.full((1,), n_sources),
                "plocs": ploc,
                "galaxy_bools": torch.ones(1, n_sources, 1),
                "galaxy_params": gparams[ii, None],
                "star_bools": torch.zeros(1, n_sources, 1),
                "star_fluxes": torch.zeros(1, n_sources, 1),
                "star_log_fluxes": torch.zeros(1, n_sources, 1),
            }
            full_cat = FullCatalog(slen, slen, d)
            image, _, _, _, bg = blends_ds.get_images(full_cat)
            images[ii] = image
            background[ii] = bg

        # predictions from encoder
        tile_est = encoder.variational_mode(images, background)
        recon = decoder.render_images(tile_est)
        tile_est.set_all_fluxes_and_mags(decoder)
        tile_est = tile_est.cpu()
        recon = recon.detach().cpu() + background
        residuals = (images - recon) / recon.sqrt()

        # now we need to obtain flux, pred. ploc, prob. of detection in tile and std. of ploc
        # for each source
        params = {
            "images": images,
            "recon": recon,
            "resid": residuals,
            "seps": seps,
            "truth": {
                "flux": torch.tensor([flux1, flux2]).reshape(1, 2, 1).expand(batch_size, 2, 1),
                "ploc": plocs,
            },
            "est": {
                "prob_n_source": torch.zeros(batch_size, 2, 1),
                "flux": torch.zeros(batch_size, 2, 1),
                "ploc": torch.zeros(batch_size, 2, 2),
                "ploc_sd": torch.zeros(batch_size, 2, 2),
            },
            "tile_est": tile_est.to_dict(),
        }
        for ii, sep in enumerate(seps):
            # get tile_est for a single batch
            d = tile_est.to_dict()
            d = {k: v[ii, None] for k, v in d.items()}
            tile_est_ii = TileCatalog(tile_slen, d)

            ploc = plocs[ii]
            params_at_coord = tile_est_ii.get_tile_params_at_coord(ploc)
            prob_n_source = torch.exp(params_at_coord["n_source_log_probs"])
            flux = params_at_coord["fluxes"]
            ploc_sd = params_at_coord["loc_sd"] * tile_slen
            loc = params_at_coord["locs"]
            assert prob_n_source.shape == flux.shape == (2, 1)
            assert ploc_sd.shape == loc.shape == (2, 2)

            if sep < 2:
                params["est"]["prob_n_source"][ii][0] = prob_n_source[0]
                params["est"]["flux"][ii][0] = flux[0]
                params["est"]["ploc"][ii][0] = loc[0] * tile_slen + 5 * tile_slen
                params["est"]["ploc_sd"][ii][0] = ploc_sd[0]

                params["est"]["prob_n_source"][ii][1] = torch.nan
                params["est"]["flux"][ii][1] = torch.nan
                params["est"]["ploc"][ii][1] = torch.tensor([torch.nan, torch.nan])
                params["est"]["ploc_sd"][ii][1] = torch.tensor([torch.nan, torch.nan])
            else:
                bias = 5 + np.ceil((sep - 2) / 4)
                params["est"]["prob_n_source"][ii] = prob_n_source
                params["est"]["flux"][ii] = flux
                params["est"]["ploc"][ii][0] = loc[0] * tile_slen + 5 * tile_slen
                params["est"]["ploc"][ii, 1, 0] = loc[1][0] * tile_slen + 5 * tile_slen
                params["est"]["ploc"][ii, 1, 1] = loc[1][1] * tile_slen + bias * tile_slen
                params["est"]["ploc_sd"][ii] = ploc_sd

        return params

    def create_figure(self, data) -> Figure:
        seps: np.ndarray = data["seps"]
        images: np.ndarray = data["images"]
        tplocs: np.ndarray = data["truth"]["ploc"]
        eplocs: np.ndarray = data["est"]["ploc"]

        # first, create image with 3 example separations (very blended to not blended)
        bp = 24
        fig, axes = plt.subplots(1, 3, figsize=(12, 7))
        axes = axes.flatten()
        seps_to_plot = self.separations_to_plot
        trim = 20  # zoom into relevant part of the image

        c1 = plt.rcParams["axes.prop_cycle"].by_key()["color"][1]  # true
        c2 = plt.rcParams["axes.prop_cycle"].by_key()["color"][3]  # predicted

        for ii, psep in enumerate(seps_to_plot):
            indx = list(seps).index(psep)
            image = images[indx, 0, trim:-trim, trim:-trim]
            x1 = tplocs[indx, :, 1] + bp - 0.5 - trim
            y1 = tplocs[indx, :, 0] + bp - 0.5 - trim
            x2 = eplocs[indx, :, 1] + bp - 0.5 - trim
            y2 = eplocs[indx, :, 0] + bp - 0.5 - trim
            axes[ii].imshow(image, cmap="gray")
            axes[ii].scatter(x1, y1, marker="x", color="r", s=30, label=None if ii else "Truth")
            axes[ii].scatter(x2, y2, marker="+", color="b", s=50, label=None if ii else "Predicted")
            axes[ii].set_xticks([0, 10, 20, 30, 40, 50])
            axes[ii].set_yticks([0, 10, 20, 30, 40, 50])
            axes[ii].set_title(rf"\rm Separation: {psep} pixels")

            if ii == 0:
                axes[ii].legend(loc="best", prop={"size": 14}, markerscale=2)

            if ii > 0:
                axes[ii].set_yticks([])  # turn off axis
                axes[ii].set_ylim(axes[0].get_ylim())  # align axes

            axes[ii].text(x1[0].item(), y1[0].item() - 7, "1", color=c1)
            axes[ii].text(x1[1].item(), y1[1].item() - 7, "2", color=c2)

        fig.tight_layout()

        return fig


class ToySeparationFigureResiduals(ToySeparationFigure):
    @property
    def rc_kwargs(self):
        return {"fontsize": 22}

    @property
    def cache_name(self) -> str:
        return "toy_separation"

    @property
    def name(self) -> str:
        return "toy_residuals"

    def create_figure(self, data) -> Figure:
        n_examples = 3
        seps_to_plot = [4, 8, 12]
        seps = data["seps"]
        images_all, recon_all, res_all = data["images"], data["recon"], data["resid"]
        trim = 20
        indices = np.array([list(seps).index(psep) for psep in seps_to_plot]).astype(int)

        images = images_all[indices, 0, trim:-trim, trim:-trim]
        recons = recon_all[indices, 0, trim:-trim, trim:-trim]
        residuals = res_all[indices, 0, trim:-trim, trim:-trim]

        pad = 6.0
        fig, axes = plt.subplots(nrows=n_examples, ncols=3, figsize=(11, 18))

        for i in range(n_examples):
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
            image = images[i]
            recon = recons[i]
            res = residuals[i]

            vmin = min(image.min().item(), recon.min().item())
            vmax = max(image.max().item(), recon.max().item())
            vmin_res = res.min().item()
            vmax_res = res.max().item()

            # plot images
            plot_image(fig, ax_true, image, vrange=(vmin, vmax))
            plot_image(fig, ax_recon, recon, vrange=(vmin, vmax))
            plot_image(fig, ax_res, res, vrange=(vmin_res, vmax_res))

        plt.subplots_adjust(hspace=-0.9)
        plt.tight_layout()
        return fig


class ToySeparationFigureMeasurements(ToySeparationFigure):
    @property
    def rc_kwargs(self):
        return {"fontsize": 22, "tick_label_size": "small", "legend_fontsize": "small"}

    @property
    def cache_name(self) -> str:
        return "toy_separation"

    @property
    def name(self) -> str:
        return "toy_measurements"

    def create_figure(self, data) -> Figure:
        fig, axes = plt.subplots(2, 2, figsize=(12, 12))
        axs = axes.flatten()
        seps = data["seps"]
        xticks = [sep for sep in seps if sep % 2 == 0]

        c1 = plt.rcParams["axes.prop_cycle"].by_key()["color"][1]
        c2 = plt.rcParams["axes.prop_cycle"].by_key()["color"][3]

        # probability of detection in each tile
        prob_n1 = data["est"]["prob_n_source"][:, 0]
        prob_n2 = data["est"]["prob_n_source"][:, 1]

        axs[0].plot(seps, prob_n1, "-", label="Galaxy 1", color=c1)
        axs[0].plot(seps, prob_n2, "-", label="Galaxy 2", color=c2)
        axs[0].axvline(2, color="k", ls="--", label=r"\rm Tile boundary")
        axs[0].axvline(6, ls="--", color="k")
        axs[0].axvline(10, ls="--", color="k")
        axs[0].axvline(14, ls="--", color="k")
        axs[0].legend(loc="best")
        axs[0].set_xticks(xticks)
        axs[0].set_xlim(0, 16)
        axs[0].set_xlabel(r"\rm Separation (pixels)")
        axs[0].set_ylabel(r"\rm Detection Probability")

        # distance residual
        tploc1 = data["truth"]["ploc"][:, 0]
        tploc2 = data["truth"]["ploc"][:, 1]
        eploc1 = data["est"]["ploc"][:, 0]
        eploc2 = data["est"]["ploc"][:, 1]
        dist1 = ((tploc1 - eploc1) ** 2).sum(1) ** (1 / 2)
        dist2 = ((tploc2 - eploc2) ** 2).sum(1) ** (1 / 2)

        axs[1].plot(seps, dist1, "-", color=c1)
        axs[1].plot(seps, dist2, "-", color=c2)
        axs[1].axhline(0, ls="-", color="k")
        axs[1].axvline(2, ls="--", color="k", label=r"\rm Tile boundary")
        axs[1].axvline(6, ls="--", color="k")
        axs[1].axvline(10, ls="--", color="k")
        axs[1].axvline(14, ls="--", color="k")
        axs[1].set_xticks(xticks)
        axs[1].set_xlim(0, 16)
        axs[1].set_xlabel(r"\rm Separation (pixels)")
        axs[1].set_ylabel(r"\rm Distance residual (pixels)")

        # location error (squared sum) estimate
        eploc_sd1 = (data["est"]["ploc_sd"][:, 0] ** 2).sum(1) ** (1 / 2)
        eploc_sd2 = (data["est"]["ploc_sd"][:, 1] ** 2).sum(1) ** (1 / 2)
        axs[3].plot(seps, eploc_sd1, "-", color=c1)
        axs[3].plot(seps, eploc_sd2, "-", color=c2)
        axs[3].axhline(0, ls="-", color="k")
        axs[3].axvline(2, ls="--", color="k", label=r"\rm Tile boundary")
        axs[3].axvline(6, ls="--", color="k")
        axs[3].axvline(10, ls="--", color="k")
        axs[3].axvline(14, ls="--", color="k")
        axs[3].set_xticks(xticks)
        axs[3].set_xlim(0, 16)
        axs[3].set_ylim(axs[1].get_ylim())
        axs[3].set_xlabel(r"\rm Separation (pixels)")
        axs[3].set_ylabel(r"\rm Predicted centroid std. (pixels)")

        # flux normalized residuals
        tflux1 = data["truth"]["flux"][:, 0]
        tflux2 = data["truth"]["flux"][:, 1]
        eflux1 = data["est"]["flux"][:, 0]
        eflux2 = data["est"]["flux"][:, 1]
        rflux1 = (tflux1 - eflux1) / tflux1
        rflux2 = (tflux2 - eflux2) / tflux2

        axs[2].plot(seps, rflux1, "-", color=c1)
        axs[2].plot(seps, rflux2, "-", color=c2)
        axs[2].axhline(0, ls="-", color="k")
        axs[2].axvline(2, ls="--", color="k", label=r"\rm Tile boundary")
        axs[2].axvline(6, ls="--", color="k")
        axs[2].axvline(10, ls="--", color="k")
        axs[2].axvline(14, ls="--", color="k")
        axs[2].set_xticks(xticks)
        axs[2].set_xlim(0, 16)
        axs[2].set_xlabel(r"\rm Separation (pixels)")
        axs[2].set_ylabel(r"\rm $(f^{\rm true} - f^{\rm recon}) / f^{\rm true}$")

        return fig


def _load_models(cfg, device):
    # load models required for SDSS reconstructions.

    location = instantiate(cfg.models.detection_encoder).to(device).eval()
    location.load_state_dict(
        torch.load(cfg.plots.location_checkpoint, map_location=location.device)
    )

    binary = instantiate(cfg.models.binary).to(device).eval()
    binary.load_state_dict(torch.load(cfg.plots.binary_checkpoint, map_location=binary.device))

    galaxy = instantiate(cfg.models.galaxy_encoder).to(device).eval()
    galaxy.load_state_dict(torch.load(cfg.plots.galaxy_checkpoint, map_location=galaxy.device))

    n_images_per_batch = cfg.plots.encoder.n_images_per_batch
    n_rows_per_batch = cfg.plots.encoder.n_rows_per_batch
    encoder = Encoder(
        location.eval(),
        binary.eval(),
        galaxy.eval(),
        n_images_per_batch=n_images_per_batch,
        n_rows_per_batch=n_rows_per_batch,
    )
    encoder = encoder.to(device)
    decoder: ImageDecoder = instantiate(cfg.models.decoder).to(device).eval()
    return encoder, decoder


def _setup(cfg):
    pcfg = cfg.plots
    figs = set(pcfg.figs)
    cachedir = pcfg.cachedir
    device = torch.device(pcfg.device)
    bfig_kwargs = {
        "figdir": pcfg.figdir,
        "cachedir": cachedir,
        "img_format": pcfg.image_format,
    }

    if not Path(cachedir).exists():
        warnings.warn("Specified cache directory does not exist, will attempt to create it.")
        Path(cachedir).mkdir(exist_ok=True, parents=True)

    assert set(figs).issubset(set(ALL_FIGS))
    return figs, device, bfig_kwargs


def _make_autoencoder_figures(cfg, device, overwrite: bool, bfig_kwargs: dict):
    print("INFO: Creating autoencoder figures...")
    autoencoder = instantiate(cfg.models.galaxy_net)
    autoencoder.load_state_dict(torch.load(cfg.models.prior.galaxy_prior.autoencoder_ckpt))
    autoencoder = autoencoder.to(device).eval()

    # generate galsim simulated galaxies images if file does not exist.
    galaxies_file = Path(cfg.plots.sim_single_gals_file)
    if not galaxies_file.exists() or overwrite:
        print(f"INFO: Generating individual galaxy images and saving to: {galaxies_file}")
        dataset = instantiate(
            cfg.datasets.sdss_galaxies, batch_size=512, n_batches=20, num_workers=20
        )
        imagepath = galaxies_file.parent / (galaxies_file.stem + "_images.png")
        generate.generate(
            dataset, galaxies_file, imagepath, n_plots=25, global_params=("background", "slen")
        )

    # arguments for figures
    psf_params_file = cfg.plots.psf_params_file
    sdss_pixel_scale = cfg.plots.sdss_pixel_scale
    args = (autoencoder, galaxies_file, psf_params_file, sdss_pixel_scale)

    # create figure classes and plot.
    AutoEncoderReconRandom(n_examples=5, overwrite=overwrite, **bfig_kwargs)(*args)
    AutoEncoderReconWorst(n_examples=5, overwrite=False, **bfig_kwargs)(*args)
    AutoEncoderBinMeasurements(overwrite=False, **bfig_kwargs)(*args)
    AutoEncoderHistograms(overwrite=False, **bfig_kwargs)(*args)
    mpl.rc_file_defaults()


def _make_blend_figures(cfg, encoder, decoder, overwrite: bool, bfig_kwargs: dict):
    print("INFO: Creating figures for metrics on simulated blended galaxies.")
    blend_file = Path(cfg.plots.sim_blend_gals_file)

    # create dataset of blends if not existant.
    if not blend_file.exists() or cfg.plots.overwrite:
        print(f"INFO: Creating dataset of simulated galsim blends and saving to {blend_file}")
        dataset = instantiate(cfg.plots.galsim_blends)
        imagepath = blend_file.parent / (blend_file.stem + "_images.png")
        global_params = ("background", "slen")
        generate.generate(dataset, blend_file, imagepath, n_plots=25, global_params=global_params)

    BlendResidualFigure(overwrite=overwrite, **bfig_kwargs)(blend_file, encoder, decoder)
    BlendDetectionFigure(overwrite=False, **bfig_kwargs)(blend_file, encoder, decoder)
    BlendHistogramFigure(overwrite=False, **bfig_kwargs)(blend_file, encoder, decoder)
    BlendClassificationFigure(overwrite=False, **bfig_kwargs)(blend_file, encoder, decoder)


@hydra.main(config_path="./config", config_name="config", version_base=None)
def main(cfg):
    figs, device, bfig_kwargs = _setup(cfg)
    encoder, decoder = _load_models(cfg, device)
    overwrite = cfg.plots.overwrite

    # FIGURE 1: Autoencoder single galaxy reconstruction
    if "single_gal" in figs:
        _make_autoencoder_figures(cfg, device, overwrite, bfig_kwargs)

    if "blend_gal" in figs:
        _make_blend_figures(cfg, encoder, decoder, overwrite, bfig_kwargs)

    if "toy" in figs:
        print("INFO: Creating figures for testing BLISS on pair galaxy toy example.")
        blend_ds = instantiate(cfg.plots.galsim_blends)
        ToySeparationFigure(overwrite=overwrite, **bfig_kwargs)(encoder, decoder, blend_ds)
        ToySeparationFigureMeasurements(overwrite=False, **bfig_kwargs)(encoder, decoder, blend_ds)
        ToySeparationFigureResiduals(overwrite=False, **bfig_kwargs)(encoder, decoder, blend_ds)


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
