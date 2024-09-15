import matplotlib.pyplot as plt
import numpy as np
import torch
from einops import rearrange, repeat
from matplotlib.figure import Figure
from torch import Tensor
from tqdm import tqdm

from bliss.catalog import FullCatalog
from bliss.datasets.lsst import PIXEL_SCALE
from bliss.encoders.autoencoder import CenteredGalaxyDecoder
from bliss.encoders.encoder import Encoder
from bliss.plotting import BlissFigure, scatter_shade_plot
from bliss.reporting import (
    compute_bin_metrics,
    get_blendedness,
    get_boostrap_precision_and_recall,
    get_single_galaxy_measurements,
    match_by_locs,
)


class BlendSimulationFigure(BlissFigure):
    @property
    def all_rcs(self) -> dict:
        return {
            "blendsim_gal_meas": {},
            "blendsim_detection": {"fontsize": 32},
            "blendsim_classification": {"fontsize": 28},
            "blendsim_hists": {"fontsize": 28},
        }

    @property
    def cache_name(self) -> str:
        return "blendsim"

    @property
    def fignames(self) -> tuple[str, ...]:
        return (
            "blendsim_gal_meas",
            "blendsim_detection",
            "blendsim_classification",
            "blendsim_hists",
        )

    def compute_data(self, blend_file: str, encoder: Encoder, decoder: CenteredGalaxyDecoder):
        blend_data: dict[str, Tensor] = torch.load(blend_file)
        images = blend_data.pop("images").float()
        background = blend_data.pop("background").float()
        uncentered_sources = blend_data.pop("uncentered_sources").float()
        centered_sources = blend_data.pop("centered_sources").float()
        blend_data.pop("noiseless")
        blend_data.pop("paddings")

        n_batches, _, size, _ = images.shape
        assert background.shape == images.shape

        # obtain `FullCatalog` from saved data
        slen = size - 2 * (encoder.detection_encoder.bp)
        truth = FullCatalog(slen, slen, blend_data)

        # get additional truth information needed
        b, ms1, _ = truth.plocs.shape
        assert uncentered_sources.shape == (b, ms1, 1, size, size)
        assert centered_sources.shape == (b, ms1, 1, size, size)
        flat_indiv = rearrange(centered_sources, "b ms c h w -> (b ms) c h w")
        flat_bg1 = repeat(background, "b c h w -> (b ms) c h w", ms=ms1, h=size, w=size)
        tflux, tsnr, tellip = get_single_galaxy_measurements(
            flat_indiv, flat_bg1, PIXEL_SCALE, no_bar=False
        )
        truth["galaxy_fluxes"] = rearrange(tflux, "(b ms) -> b ms 1", ms=ms1)
        truth["snr"] = rearrange(tsnr, "(b ms) -> b ms 1", ms=ms1)
        truth["ellips"] = rearrange(tellip, "(b ms) g -> b ms g", ms=ms1, g=2)
        blendedness = get_blendedness(uncentered_sources)
        assert not blendedness.isnan().any()
        truth["blendedness"] = rearrange(blendedness, "b ms -> b ms 1", b=b, ms=ms1)

        # get additional galaxy quantities from predicted information
        print("INFO: BLISS posterior inference on images.")
        tile_est = encoder.variational_mode(images, background).to(torch.device("cpu"))
        est = tile_est.to_full_params()  # no need to compute galaxy quantities in tiles anymore!

        flat_galaxy_params = rearrange(est["galaxy_params"], "b ms d -> (b ms) d")
        flat_galaxy_bools = rearrange(est["galaxy_bools"], "b ms 1 -> (b ms) 1")

        # not sure if batches are necessary here
        b, ms2, _ = est["galaxy_params"].shape
        n_total = b * ms2
        flat_bg2 = repeat(
            torch.tensor([background[0, 0, 0, 0].item()]),
            "() -> (b ms) c h w",
            b=b,
            ms=ms2,
            c=1,
            h=decoder.slen,
            w=decoder.slen,
        )

        eflux = torch.zeros((n_total, 1))
        esnr = torch.zeros((n_total, 1))
        eellips = torch.zeros((n_total, 2))
        n_parts = 1000  # so it all fits in GPU

        desc = "Computing galaxy properties of predicted, reconstructed galaxies"
        for n1 in tqdm(range(0, n_total, n_parts), desc=desc, total=n_total // n_parts):

            n2 = n1 + n_parts
            galaxy_params_ii = flat_galaxy_params[n1:n2]
            galaxy_bools_ii = flat_galaxy_bools[n1:n2].cpu()
            bg_ii = flat_bg2[n1:n2]

            egals_ii_raw = decoder(galaxy_params_ii.to(encoder.device)).cpu()
            egals_ii = egals_ii_raw * rearrange(galaxy_bools_ii, "npt 1 -> npt 1 1 1")
            eflux_ii, esnr_ii, eellip_ii = get_single_galaxy_measurements(egals_ii, bg_ii)

            eflux[n1:n2, 0] = eflux_ii
            esnr[n1:n2, 0] = esnr_ii
            eellips[n1:n2, :] = eellip_ii

        # finally put into TileCatalog and get FullCatalog
        est["fluxes"] = rearrange(eflux, "(b ms) 1 -> b ms 1", b=b, ms=ms2)
        est["snr"] = rearrange(esnr, "(b ms) 1 -> b ms 1", b=b, ms=ms2)
        est["ellips"] = rearrange(eellips, "(b ms) g -> b ms g", b=b, ms=ms2, g=2)

        # compute detection metrics (snr)
        snr_bins2 = 10 ** torch.arange(0.2, 3.2, 0.2)
        snr_bins1 = 10 ** torch.arange(0, 3.0, 0.2)
        snr_bins = torch.column_stack((snr_bins1, snr_bins2))
        bin_metrics = compute_bin_metrics(truth, est, "snr", snr_bins)
        boot_metrics = get_boostrap_precision_and_recall(1000, truth, est, "snr", snr_bins)

        # collect quantities for residuals on ellipticites and flux of galaxies
        snr = []
        blendedness = []
        true_fluxes = []
        true_ellips1 = []
        true_ellips2 = []
        est_fluxes = []
        est_ellips1 = []
        est_ellips2 = []
        snr_class = []
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
                gbool_ii = torch.logical_and(torch.eq(tgbool_ii, egbool_ii), tgbool_ii)
                gbool_ii = gbool_ii.flatten()
                snr_ii_class = truth["snr"][ii][tindx][dkeep]

                assert len(tgbool_ii) == len(egbool_ii) == len(snr_ii_class) == n_matches

                # save snr, mag, and booleans over matches for classification metrics
                for jj in range(n_matches):
                    snr_class.append(snr_ii_class[jj].item())
                    tgbools.append(tgbool_ii[jj].item())
                    egbools.append(egbool_ii[jj].item())

                snr_ii = truth["snr"][ii][tindx][dkeep][gbool_ii]  # noqa: WPS219
                blendedness_ii = truth["blendedness"][ii][tindx][dkeep][gbool_ii]  # noqa: WPS219
                true_flux_ii = truth["fluxes"][ii][tindx][dkeep][gbool_ii]  # noqa: WPS219
                est_flux_ii = est["fluxes"][ii][eindx][dkeep][gbool_ii]  # noqa: WPS219
                true_ellips_ii = truth["ellips"][ii][tindx][dkeep][gbool_ii]  # noqa: WPS219
                est_ellips_ii = est["ellips"][ii][eindx][dkeep][gbool_ii]  # noqa: WPS219

                n_matched_gals = len(snr_ii)

                for kk in range(n_matched_gals):
                    snr.append(snr_ii[kk].item())
                    blendedness.append(blendedness_ii[kk].item())
                    true_ellips1.append(true_ellips_ii[kk][0].item())
                    true_ellips2.append(true_ellips_ii[kk][1].item())
                    est_ellips1.append(est_ellips_ii[kk][0].item())
                    est_ellips2.append(est_ellips_ii[kk][1].item())
                    true_fluxes.append(true_flux_ii[kk].item())
                    est_fluxes.append(est_flux_ii[kk].item())

        true_ellips = torch.vstack([torch.tensor(true_ellips1), torch.tensor(true_ellips2)])
        true_ellips = true_ellips.T.reshape(-1, 2)

        est_ellips = torch.vstack([torch.tensor(est_ellips1), torch.tensor(est_ellips2)])
        est_ellips = est_ellips.T.reshape(-1, 2)

        return {
            "residuals": {
                "snr": torch.tensor(snr),
                "blendedness": torch.tensor(blendedness),
                "true_fluxes": torch.tensor(true_fluxes),
                "est_fluxes": torch.tensor(est_fluxes),
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
                "tgbools": torch.tensor(tgbools),
                "egbools": torch.tensor(egbools),
            },
        }

    def _get_residual_blend_figure(self, data) -> Figure:
        snr, blendedness, tfluxes, efluxes, true_ellips, est_ellips = data["residuals"].values()
        fig, axes = plt.subplots(3, 2, figsize=(12, 18), sharex="col", sharey="row")
        ax1, ax2, ax3, ax4, ax5, ax6 = axes.flatten()

        xlims = (0.5, 3)
        xlabel = r"$\log_{10} \rm SNR$"
        ylabel = r"\rm $(f^{\rm recon} - f^{\rm true}) / f^{\rm true}$"
        x, y = np.log10(snr), (efluxes - tfluxes) / tfluxes
        scatter_shade_plot(ax1, x, y, xlims, delta=0.2, use_boot=True)
        ax1.set_ylabel(ylabel)
        ax1.axhline(0, ls="--", color="k")
        ax1.set_ylim(-0.2, 0.4)

        xlims = (0, 0.5)
        x, y = blendedness, (efluxes - tfluxes) / tfluxes
        scatter_shade_plot(ax2, x, y, xlims, delta=0.05, use_boot=True)
        ax2.axhline(0, ls="--", color="k")
        ax2.set_ylim(-0.2, 0.4)

        # need to mask the (very few) ellipticities that are NaNs from adaptive moments
        te1, te2 = true_ellips[:, 0], true_ellips[:, 1]
        pe1, pe2 = est_ellips[:, 0], est_ellips[:, 1]
        mask1 = np.isnan(te1)
        mask2 = np.isnan(pe1)
        mask = ~mask1 & ~mask2  # only need one component bc how func is written
        print(f"INFO: Total number of true ellipticity NaNs is: {sum(mask1)}")
        print(f"INFO: Total number of reconstructed ellipticity NaNs is: {sum(mask2)}")
        xlims = (0.5, 3)
        ylabel = r"$g_{1}^{\rm recon} - g_{1}^{\rm true}$"
        x, y = np.log10(snr)[mask], (pe1 - te1)[mask]
        scatter_shade_plot(ax3, x, y, xlims, delta=0.2, use_boot=True)
        ax3.set_ylabel(ylabel)
        ax3.axhline(0, ls="--", color="k")
        ax3.set_ylim(-0.05, 0.1)

        xlims = (0, 0.5)
        x, y = blendedness[mask], (pe1 - te1)[mask]
        scatter_shade_plot(ax4, x, y, xlims, delta=0.05, use_boot=True)
        ax4.axhline(0, ls="--", color="k")
        ax4.set_ylim(-0.05, 0.1)

        xticks = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
        xlims = (0.5, 3)
        xlabel = r"$\log_{10} \rm SNR$"
        ylabel = r"$g_{2}^{\rm recon} - g_{2}^{\rm true}$"
        x, y = np.log10(snr)[mask], (pe2 - te2)[mask]
        scatter_shade_plot(ax5, x, y, xlims, delta=0.2, use_boot=True)
        ax5.set_xlabel(xlabel)
        ax5.set_ylabel(ylabel)
        ax5.set_xticks(xticks)
        ax5.axhline(0, ls="--", color="k")
        ax5.set_ylim(-0.05, 0.1)

        xticks = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
        xlims = (0, 0.5)
        xlabel = "$B$"
        x, y = blendedness[mask], (pe2 - te2)[mask]
        scatter_shade_plot(ax6, x, y, xlims=xlims, delta=0.05, use_boot=True)
        ax6.set_xlabel(xlabel)
        ax6.set_xticks(xticks)
        ax6.axhline(0, ls="--", color="k")
        ax6.set_ylim(-0.05, 0.1)

        plt.tight_layout()

        return fig

    def _get_detection_figure(self, data) -> Figure:
        # take middle of bin as x for plotting
        snr_bins = np.log10(data["detection"]["bins"].mean(1))
        return _make_pr_figure(
            snr_bins, data["detection"], r"$\log_{10} \rm SNR$", xlims=(0.5, 3), ylims2=(0, 2000)
        )

    def _compute_pr_class(self, tgbool: np.ndarray, egbool: np.ndarray):
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

    def _get_classification_figure(self, data) -> Figure:
        snrs, tgbools, egbools = data["classification"].values()
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

                p, r = self._compute_pr_class(tgbool_ii, egbool_ii)
                boot_precision[ii][jj] = p
                boot_recall[ii][jj] = r

        # compute precision and recall per bin
        for jj, (b1, b2) in enumerate(snr_bins):
            keep = (b1 < snrs) & (snrs < b2)
            tgbool = tgbools[keep]
            egbool = egbools[keep]
            p, r = self._compute_pr_class(tgbool, egbool)
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

    def _get_histogram_figure(self, data) -> Figure:
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

    def create_figure(self, fname: str, data) -> Figure:
        if fname == "blendsim_gal_meas":
            return self._get_residual_blend_figure(data)
        if fname == "blendsim_detection":
            return self._get_detection_figure(data)
        if fname == "blendsim_classification":
            return self._get_classification_figure(data)
        if fname == "blendsim_hists":
            return self._get_histogram_figure(data)
        raise NotImplementedError("Figure {fname} not implemented.")


def _make_pr_figure(
    bins: np.ndarray,
    data: dict[str, np.ndarray],
    xlabel: str,
    xlims: tuple[float, float] = None,
    ylims: tuple[float, float] = None,
    ylims2: tuple[float, float] = None,
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
    mask1 = ~np.isnan(precision)
    mask2 = ~np.isnan(recall)
    ymin = min(precision[mask1].min(), recall[mask2].min())
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

    ax2.legend()
    ax2.set_xlabel(xlabel)
    ax2.set_ylabel(rf"\rm {metric_type} Metric")
    ax2.set_yticks(yticks)
    ax2.grid(linestyle="-", linewidth=0.5, which="major", axis="both")

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
    ymax = max(tgcount.max(), tscount.max(), egcount.max(), escount.max())
    yticks = np.round(np.linspace(0, ymax, n_ticks), -ordmag)
    ax1.set_yticks(yticks)
    ax1.set_ylabel(r"\rm Counts")
    ax1.legend(loc="best", prop={"size": legend_size_hist})
    ax1.grid(linestyle="-", linewidth=0.5, which="major", axis="both")
    plt.subplots_adjust(hspace=0)
    return fig
