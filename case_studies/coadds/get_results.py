#!/usr/bin/env python3
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import hydra
import numpy as np
import torch
from hydra.utils import instantiate
from matplotlib import pyplot as plt
from torch import Tensor
from tqdm import tqdm

from bliss.catalog import FullCatalog
from bliss.datasets.sdss import convert_flux_to_mag
from bliss.encoder import Encoder
from bliss.reporting import DetectionMetrics, match_by_locs
from case_studies.coadds.coadd_decoder import load_coadd_dataset
from case_studies.sdss_galaxies.plots.bliss_figures import CB_color_cycle, set_rc_params

device = torch.device("cuda:0")

latex_names = {
    "single": r"\rm single-exposure",
    "coadd_5": r"\rm Coadd $d=5$",
    "coadd_10": r"\rm Coadd $d=10$",
    "coadd_25": r"\rm Coadd $d=25$",
    "coadd_35": r"\rm Coadd $d=35$",
    "coadd_50": r"\rm Coadd $d=50$",
}

set_rc_params(fontsize=28)


def scatter_shade_plot(ax, x, y, xlims, delta, qs=(0.25, 0.75), color="m"):
    # plot median and 25/75 quantiles on each bin decided by delta and xlims.

    xbins = np.arange(xlims[0], xlims[1], delta)

    xs = np.zeros(len(xbins))
    ys = np.zeros(len(xbins))
    yqs = np.zeros((len(xbins), 2))

    for i, bx in enumerate(xbins):
        keep_x = (x > bx) & (x < bx + delta)
        y_bin = y[keep_x]

        xs[i] = bx + delta / 2

        if len(y_bin) == 0:  # noqa: WPS507
            ys[i] = np.nan
            yqs[i] = (np.nan, np.nan)
            continue

        ys[i] = np.median(y_bin)
        yqs[i, :] = np.quantile(y_bin, qs[0]), np.quantile(y_bin, qs[1])

    ax.plot(xs, ys, marker="o", c=color, linestyle="-")
    ax.fill_between(xs, yqs[:, 0], yqs[:, 1], color=color, alpha=0.5)


def _compute_mag_bin_metrics(
    mag_bins: Tensor, truth: FullCatalog, est: FullCatalog
) -> Dict[str, Tensor]:
    metrics_per_mag = defaultdict(lambda: torch.zeros(len(mag_bins)))

    # compute data for precision/recall/classification accuracy as a function of magnitude.
    for ii, (mag1, mag2) in tqdm(enumerate(mag_bins), desc="Metrics per bin", total=len(mag_bins)):
        detection_metrics = DetectionMetrics(disable_bar=True)

        # precision
        eparams = est.apply_param_bin("mags", mag1, mag2)
        detection_metrics.update(truth, eparams)
        precision = detection_metrics.compute()["precision"]
        detection_metrics.reset()

        # recall
        tparams = truth.apply_param_bin("mags", mag1, mag2)
        detection_metrics.update(tparams, est)
        recall = detection_metrics.compute()["recall"]

        tcount = tparams.n_sources.sum().item()
        ecount = eparams.n_sources.sum().item()

        metrics_per_mag["precision"][ii] = precision
        metrics_per_mag["recall"][ii] = recall
        metrics_per_mag["tcount"][ii] = tcount
        metrics_per_mag["ecount"][ii] = ecount

    return dict(metrics_per_mag)


def _compute_tp_fp(truth: FullCatalog, est: FullCatalog):
    # get precision tp per batch
    all_tp = torch.zeros(truth.batch_size)
    all_fp = torch.zeros(truth.batch_size)
    all_ntrue = torch.zeros(truth.batch_size)
    for b in range(truth.batch_size):
        ntrue, nest = truth.n_sources[b].int().item(), est.n_sources[b].int().item()
        tlocs, elocs = truth.plocs[b], est.plocs[b]
        if ntrue > 0 and nest > 0:
            _, mest, dkeep, _ = match_by_locs(tlocs, elocs)
            tp = len(elocs[mest][dkeep])  # n_matches
            fp = nest - tp
        elif ntrue > 0:
            tp = 0
            fp = 0
        elif nest > 0:
            tp = 0
            fp = nest
        else:
            tp = 0
            fp = 0
        all_tp[b] = tp
        all_fp[b] = fp
        all_ntrue[b] = ntrue

    return all_tp, all_fp, all_ntrue


def _comput_tp_fp_per_bin(mag_bins: Tensor, truth: FullCatalog, est: FullCatalog):
    counts_per_mag = defaultdict(lambda: torch.zeros(len(mag_bins), truth.batch_size))
    for ii, (mag1, mag2) in tqdm(enumerate(mag_bins), desc="tp/fp per bin", total=len(mag_bins)):

        # precision
        eparams = est.apply_param_bin("mags", mag1, mag2)
        tp, fp, ntrue = _compute_tp_fp(truth, eparams)
        counts_per_mag["tp_precision"][ii] = tp
        counts_per_mag["fp_precision"][ii] = fp

        # recall
        tparams = truth.apply_param_bin("mags", mag1, mag2)
        tp, _, ntrue = _compute_tp_fp(tparams, est)
        counts_per_mag["tp_recall"][ii] = tp
        counts_per_mag["ntrue"][ii] = ntrue

    return counts_per_mag


def _get_bootstrap_pr_err(n_samples: int, mag_bins: Tensor, truth: FullCatalog, est: FullCatalog):
    """Get errors for precision/recall which need to be handled carefully to be efficient."""
    counts_per_mag = _comput_tp_fp_per_bin(mag_bins, truth, est)
    batch_size = truth.batch_size
    n_mag_bins = mag_bins.shape[0]

    # get counts in needed format
    tpp_boot = counts_per_mag["tp_precision"].unsqueeze(0).expand(n_samples, n_mag_bins, batch_size)
    fpp_boot = counts_per_mag["fp_precision"].unsqueeze(0).expand(n_samples, n_mag_bins, batch_size)
    tpr_boot = counts_per_mag["tp_recall"].unsqueeze(0).expand(n_samples, n_mag_bins, batch_size)
    ntrue_boot = counts_per_mag["ntrue"].unsqueeze(0).expand(n_samples, n_mag_bins, batch_size)

    # get indices to boostrap
    # NOTE: the indices for each sample repeat across magnitude bins
    boot_indices = torch.randint(0, batch_size, (n_samples, 1, batch_size))
    boot_indices = boot_indices.expand(n_samples, n_mag_bins, batch_size)

    # get bootstrapped samples of counts
    tpp_boot = torch.gather(tpp_boot, 2, boot_indices)
    fpp_boot = torch.gather(fpp_boot, 2, boot_indices)
    tpr_boot = torch.gather(tpr_boot, 2, boot_indices)
    ntrue_boot = torch.gather(ntrue_boot, 2, boot_indices)
    assert tpp_boot.shape == (n_samples, n_mag_bins, batch_size)

    # finally, get precision and recall boostrapped samples
    precision_boot = tpp_boot.sum(2) / (tpp_boot.sum(2) + fpp_boot.sum(2))
    recall_boot = tpr_boot.sum(2) / ntrue_boot.sum(2)

    return {"precision": precision_boot, "recall": recall_boot}


def _load_encoder(cfg, model_path) -> Encoder:
    model = instantiate(cfg.models.detection_encoder).to(device).eval()
    model.load_state_dict(torch.load(model_path, map_location=device))
    return Encoder(model.eval(), n_images_per_batch=10, n_rows_per_batch=15).to(device).eval()


def _run_model_on_images(encoder, images, background) -> FullCatalog:
    tile_est = encoder.variational_mode(images, background)
    return tile_est.cpu().to_full_params()


def _add_mags_to_catalog(cat: FullCatalog) -> FullCatalog:
    batch_size = cat.n_sources.shape[0]
    cat["mags"] = torch.zeros_like(cat["star_fluxes"])
    for ii in range(batch_size):
        n_sources = cat.n_sources[ii].item()
        cat["mags"][ii, :n_sources] = convert_flux_to_mag(cat["star_fluxes"][ii, :n_sources])
    return cat


def _get_matched_fluxes(truth: FullCatalog, est: FullCatalog):
    tfluxes = []
    efluxes = []
    log_tfluxes = []
    log_efluxes = []
    log_sd_efluxes = []

    batch_size = truth.batch_size
    for ii in tqdm(range(batch_size), desc="computing matched fluxes"):
        n_sources1, n_sources2 = truth.n_sources[ii].item(), est.n_sources[ii].item()
        if n_sources1 > 0 and n_sources2 > 0:
            plocs1 = truth.plocs[ii]
            plocs2 = est.plocs[ii]
            mtrue, mest, dkeep, _ = match_by_locs(plocs1, plocs2)
            tp = len(plocs2[mest][dkeep])  # n_matches
            fluxes1 = truth["star_fluxes"][ii][mtrue][dkeep]
            fluxes2 = est["star_fluxes"][ii][mest][dkeep]
            log_fluxes1 = truth["star_log_fluxes"][ii][mtrue][dkeep]
            log_fluxes2 = est["star_log_fluxes"][ii][mest][dkeep]
            sd_log_fluxes2 = est["log_flux_sd"][ii][mest][dkeep]
            for jj in range(tp):
                flux1 = fluxes1[jj].item()
                flux2 = fluxes2[jj].item()
                log_flux1 = log_fluxes1[jj].item()
                log_flux2 = log_fluxes2[jj].item()
                sd_log_flux2 = sd_log_fluxes2[jj].item()
                assert flux1 > 0 and flux2 > 0
                tfluxes.append(flux1)
                efluxes.append(flux2)
                log_tfluxes.append(log_flux1)
                log_efluxes.append(log_flux2)
                log_sd_efluxes.append(sd_log_flux2)

    return {
        "true_fluxes": torch.tensor(tfluxes),
        "est_fluxes": torch.tensor(efluxes),
        "true_log_fluxes": torch.tensor(log_tfluxes),
        "est_log_fluxes": torch.tensor(log_efluxes),
        "est_sd_log_fluxes": torch.tensor(log_sd_efluxes),
    }


def _create_coadd_example_plot(test_images: Tensor):

    n = 1714
    image1 = test_images["single"][n, 0]
    image10 = test_images["coadd_10"][n, 0]
    image25 = test_images["coadd_25"][n, 0]
    image50 = test_images["coadd_50"][n, 0]

    fig, axs = plt.subplots(1, 4, figsize=(24, 8))
    axs[0].imshow(image1, interpolation=None)
    axs[0].set_xlabel(xlabel="Single Exposures")
    axs[1].imshow(image10, interpolation=None)
    axs[1].set_xlabel(xlabel="$d = 10$")
    axs[2].imshow(image25, interpolation=None)
    axs[2].set_xlabel(xlabel="$d = 25$")
    im = axs[3].imshow(image50, interpolation=None)
    axs[3].set_xlabel(xlabel="$d = 50$")
    fig.colorbar(im, ax=axs.ravel().tolist())
    return fig


def _report_posterior_flux_calibration(report_file: str, model_names: list, data: dict):
    sigmas = [1, 2, 3]
    cis = [0.68, 0.95, 0.99]
    with open(report_file, "w", encoding="utf-8") as f:
        for model_name in model_names:
            matched_fluxes = data[model_name]["matched_fluxes"]
            tlfluxes = matched_fluxes["true_log_fluxes"]
            elfluxes = matched_fluxes["est_log_fluxes"]
            sd_elfluxes = matched_fluxes["est_sd_log_fluxes"]
            print(f"Model '{model_name}' posterior calibration results:", file=f)
            for s, ci in zip(sigmas, cis):
                counts = tlfluxes < elfluxes + s * sd_elfluxes
                counts &= tlfluxes > elfluxes - s * sd_elfluxes
                fraction = counts.float().mean()
                print(f"For {s}-sigma ({ci}): {fraction}", file=f)
            print(file=f)


def _create_money_plot(mag_bins: Tensor, model_names: List[str], data: dict):

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 7))

    x = (mag_bins[:, 1] + mag_bins[:, 0]) / 2

    for ii, mname in enumerate(model_names):
        color = CB_color_cycle[ii]

        # precision
        precision = data[mname]["mag_bin_metrics"]["precision"]
        boot_precision = data[mname]["boot_mag_bin_metrics"]["precision"]
        precision1 = boot_precision.quantile(0.05, 0)
        precision2 = boot_precision.quantile(0.95, 0)
        ax1.plot(x, precision, "-o", color=color, label=latex_names[mname], markersize=6)
        ax1.fill_between(x, precision1, precision2, color=color, alpha=0.5)

        # recall
        recall = data[mname]["mag_bin_metrics"]["recall"]
        boot_recall = data[mname]["boot_mag_bin_metrics"]["recall"]
        recall1 = boot_recall.quantile(0.05, 0)
        recall2 = boot_recall.quantile(0.95, 0)
        ax2.plot(x, recall, "-o", color=color, label=latex_names[mname], markersize=6)
        ax2.fill_between(x, recall1, recall2, color=color, alpha=0.5)

        # fluxes
        matched_fluxes = data[mname]["matched_fluxes"]
        tfluxes = matched_fluxes["true_fluxes"]
        efluxes = matched_fluxes["est_fluxes"]
        res_flux = (efluxes - tfluxes) / tfluxes
        tmags = convert_flux_to_mag(tfluxes)
        scatter_shade_plot(ax3, tmags, res_flux, (20, 23), delta=0.2, color=color, qs=(0.25, 0.75))

    ax1.set_xlabel(r"\rm Magnitude")
    ax2.set_xlabel(r"\rm Magnitude")
    ax1.set_ylabel(r"\rm Precision")
    ax2.set_ylabel(r"\rm Recall")
    ax3.set_xlabel(r"\rm Magnitude")
    ax3.set_ylabel(r"\rm $(f^{\rm pred} - f^{\rm true}) / f^{\rm true}$")
    ax3.axhline(0, linestyle="--", color="k")
    ax1.legend(loc="best", prop={"size": 24})
    ax1.minorticks_on()
    ax2.minorticks_on()
    ax3.minorticks_on()
    ax1.set_xlim(20, 23)
    ax2.set_xlim(20, 23)
    ax3.set_xlim(20, 23)

    return fig


@hydra.main(config_path="./config", config_name="config")
def main(cfg):

    mag1, mag2, delta = cfg.results.mag_bins
    mag_bins1 = torch.arange(mag1, mag2, delta)
    mag_bins2 = mag_bins1 + delta
    mag_bins = torch.column_stack((mag_bins1, mag_bins2))

    model_names = cfg.results.models  # e.g. 'coadd50', 'single'
    test_path = cfg.results.test_path
    all_test_images, truth = load_coadd_dataset(test_path)

    if cfg.results.overwrite:
        background_obj = instantiate(cfg.datasets.galsim_blends.background)

        outputs = {}

        for model_name in model_names:

            # run model and get catalogs
            model_path = f"{cfg.results.models_path}/{model_name}_encoder.pt"
            test_images = all_test_images[model_name]
            background = background_obj.sample(test_images.shape)
            encoder = _load_encoder(cfg, model_path)
            est = _run_model_on_images(encoder, test_images, background)
            est["galaxy_bools"] = torch.zeros(est.batch_size, est.max_sources, 1)

            # add mags to catalog
            truth = _add_mags_to_catalog(truth)
            est = _add_mags_to_catalog(est)

            # get all metrics from catalogs
            bin_metrics = _compute_mag_bin_metrics(mag_bins, truth, est)
            boot_metrics = _get_bootstrap_pr_err(10000, mag_bins, truth, est)
            matched_fluxes = _get_matched_fluxes(truth, est)

            outputs[model_name] = {
                "mag_bin_metrics": bin_metrics,
                "boot_mag_bin_metrics": boot_metrics,
                "matched_fluxes": matched_fluxes,
            }

        torch.save(outputs, cfg.results.cache_results)

    else:
        assert Path(cfg.results.cache_results).exists()

        fig_path = Path(cfg.results.figs_path)
        fig_path.mkdir(exist_ok=True)

        data = torch.load(cfg.results.cache_results)
        _report_posterior_flux_calibration(cfg.results.report_file, model_names, data)

        # create all figures
        figs = []
        figs.append(_create_coadd_example_plot(all_test_images))
        figs.append(_create_money_plot(mag_bins, model_names, data))

        for fig, name in zip(figs, cfg.results.figs):
            fig.savefig(fig_path / f"{name}.png", format="png")


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
