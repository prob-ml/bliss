from collections import defaultdict
from typing import Dict, Union

import numpy as np
import seaborn as sns
import torch
from matplotlib import pyplot as plt
from torch import Tensor

from bliss import reporting
from bliss.catalog import FullCatalog, PhotoFullCatalog
from bliss.encoder import Encoder
from bliss.inference import SDSSFrame, SimulatedFrame, reconstruct_scene_at_coordinates
from bliss.models.decoder import ImageDecoder
from case_studies.sdss_galaxies.plots.bliss_figures import (
    BlissFigures,
    CB_color_cycle,
    format_plot,
    set_rc_params,
)


def compute_mag_bin_metrics(
    mag_bins: Tensor, truth: FullCatalog, pred: FullCatalog
) -> Dict[str, Tensor]:
    metrics_per_mag = defaultdict(lambda: torch.zeros(len(mag_bins)))

    # compute data for precision/recall/classification accuracy as a function of magnitude.
    for ii, (mag1, mag2) in enumerate(mag_bins):
        res = reporting.scene_metrics(truth, pred, mag_min=mag1, mag_max=mag2, slack=1.0)
        metrics_per_mag["precision"][ii] = res["precision"]
        metrics_per_mag["recall"][ii] = res["recall"]
        metrics_per_mag["f1"][ii] = res["f1"]
        metrics_per_mag["class_acc"][ii] = res["class_acc"]
        conf_matrix = res["conf_matrix"]
        metrics_per_mag["galaxy_acc"][ii] = conf_matrix[0, 0] / conf_matrix[0, :].sum().item()
        metrics_per_mag["star_acc"][ii] = conf_matrix[1, 1] / conf_matrix[1, :].sum().item()
        for k, v in res["counts"].items():
            metrics_per_mag[k][ii] = v

    return dict(metrics_per_mag)


def make_detection_figure(
    mags,
    data,
    xlims=(18, 24),
    ylims=(0.5, 1.05),
    ratio=2,
    where_step="mid",
    n_gap=50,
):
    # precision / recall / f1 score
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


class DetectionClassificationFigures(BlissFigures):
    cache = "detect_class.pt"

    @staticmethod
    def compute_metrics(truth: FullCatalog, pred: FullCatalog):

        # prepare magnitude bins
        mag_cuts2 = torch.arange(18, 24.5, 0.25)
        mag_cuts1 = torch.full_like(mag_cuts2, fill_value=-np.inf)
        mag_cuts = torch.column_stack((mag_cuts1, mag_cuts2))

        mag_bins2 = torch.arange(18, 25, 1.0)
        mag_bins1 = mag_bins2 - 1
        mag_bins = torch.column_stack((mag_bins1, mag_bins2))

        # compute metrics
        cuts_data = compute_mag_bin_metrics(mag_cuts, truth, pred)
        bins_data = compute_mag_bin_metrics(mag_bins, truth, pred)

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

        return {
            "mag_cuts": mag_cuts2,
            "mag_bins": mag_bins2,
            "cuts_data": cuts_data,
            "bins_data": bins_data,
            "full_metrics": full_metrics,
        }

    def compute_data(
        self,
        frame: Union[SDSSFrame, SimulatedFrame],
        photo_cat: PhotoFullCatalog,
        encoder: Encoder,
        decoder: ImageDecoder,
    ):
        bp = encoder.border_padding
        h, w = bp, bp
        h_end = ((frame.image.shape[2] - 2 * bp) // 4) * 4 + bp
        w_end = ((frame.image.shape[3] - 2 * bp) // 4) * 4 + bp
        truth_params: FullCatalog = frame.get_catalog((h, h_end), (w, w_end))
        photo_catalog_at_hw = photo_cat.crop_at_coords(h, h_end, w, w_end)

        # obtain predictions from BLISS.
        _, tile_est_params = reconstruct_scene_at_coordinates(
            encoder, decoder, frame.image, frame.background, h_range=(h, h_end), w_range=(w, w_end)
        )
        tile_est_params.set_all_fluxes_and_mags(decoder)
        est_params = tile_est_params.cpu().to_full_params()

        # compute metrics with bliss vs coadd and photo (frame) vs coadd
        bliss_metrics = self.compute_metrics(truth_params, est_params)
        photo_metrics = self.compute_metrics(truth_params, photo_catalog_at_hw)

        return {"bliss_metrics": bliss_metrics, "photo_metrics": photo_metrics}

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
        tgbool = data["tgbool"].astype(bool)
        egbool = data["egbool"].astype(bool)
        tmag, egprob = data["tmag"], data["egprob"]
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
        tgbool = data["tgbool"].astype(bool)
        tmag, emag = data["tmag"], data["emag"]

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
        f1 = make_detection_figure(mag_cuts, cuts_data, ylims=(0.5, 1.03))
        f2 = self.make_classification_figure(mag_cuts, cuts_data, "cuts", ylims=(0.8, 1.03))
        f3 = make_detection_figure(
            mag_bins - 0.5, bins_data, xlims=(17, 24), ylims=(0.0, 1.05), n_gap=25
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
        bliss_figs = self.create_metrics_figures(**data["bliss_metrics"], name="bliss_sdss")
        photo_figs = self.create_metrics_figures(**data["photo_metrics"], name="photo_sdss")
        return {**bliss_figs, **photo_figs}
