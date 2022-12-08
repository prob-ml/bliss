#!/usr/bin/env python3
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import hydra
import numpy as np
import torch
from hydra.utils import instantiate
from matplotlib import pyplot as plt
from scipy import stats
from torch import Tensor
from tqdm import tqdm

from bliss.catalog import FullCatalog, TileCatalog
from bliss.datasets.sdss import convert_flux_to_mag
from bliss.encoder import Encoder
from bliss.plotting import CB_color_cycle, set_rc_params
from bliss.reporting import DetectionMetrics, match_by_locs
from case_studies.coadds.coadds import load_coadd_dataset

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


def _comput_tp_fp_per_bin(
    mag_bins: Tensor, truth: FullCatalog, est: FullCatalog
) -> Dict[str, Tensor]:
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


def _get_bootstrap_pr_err(
    n_samples: int, mag_bins: Tensor, truth: FullCatalog, est: FullCatalog
) -> Dict[str, Tensor]:
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

    assert precision_boot.shape == (n_samples, n_mag_bins)
    assert recall_boot.shape == (n_samples, n_mag_bins)
    return {"precision": precision_boot, "recall": recall_boot}


def _load_encoder(cfg, model_path) -> Encoder:
    model = instantiate(cfg.models.detection_encoder).to(device).eval()
    model.load_state_dict(torch.load(model_path, map_location=device))
    return Encoder(model.eval(), n_images_per_batch=10, n_rows_per_batch=10).to(device).eval()


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


def _get_fluxes_cond_true_detections(
    encoder: Encoder, truth: FullCatalog, images: Tensor, background: Tensor
) -> Dict[str, Tensor]:
    """Return variational parameters on fluxes conditioning on true number of stars per tile."""
    # setup
    bp = encoder.border_padding
    detection_encoder = encoder.detection_encoder
    tile_slen = detection_encoder.tile_slen
    batch_size = images.shape[0]
    n_tiles_h = (images.shape[2] - 2 * bp) // tile_slen
    n_tiles_w = (images.shape[3] - 2 * bp) // tile_slen
    ptile_loader = encoder.make_ptile_loader(images, background, n_tiles_h)
    tile_map_list: List[Dict[str, Tensor]] = []
    assert n_tiles_h == n_tiles_w == encoder.n_rows_per_batch

    # get true detections per tile
    truth_tile_catalog = truth.to_tile_params(tile_slen, 1, ignore_extra_sources=True)
    tile_n_sources = truth_tile_catalog.n_sources  # (b, n_tiles_h, n_tiles_w)
    truth_at_most_one = truth_tile_catalog.to_full_params()

    # organize `tile_n_sources` to match `ptiles` from loader.
    size = n_tiles_h**2 * encoder.n_images_per_batch
    n_ptiles_per_iter = int(batch_size * n_tiles_h**2 // size)
    ptiles_n_sources = torch.zeros(n_ptiles_per_iter, size, dtype=torch.int64)
    n_images_per_batch = encoder.n_images_per_batch
    for ii in range(0, size):
        start, end = n_images_per_batch * ii, (n_images_per_batch * (ii + 1))
        ptiles_n_sources[ii] = tile_n_sources[start:end].reshape(-1)

    with torch.no_grad():
        for ii, ptiles in enumerate(tqdm(ptile_loader, desc="Encoding ptiles")):
            dist_params = detection_encoder.encode(ptiles)
            n_sources_ii = ptiles_n_sources[ii].unsqueeze(0)
            dist_params_n_src = detection_encoder.encode_for_n_sources(
                dist_params["per_source_params"], n_sources_ii
            )
            tile_samples = {}
            tile_samples["n_sources"] = n_sources_ii.cpu()
            tile_samples["locs"] = dist_params_n_src["loc_mean"].cpu()
            tile_samples["star_log_fluxes"] = dist_params_n_src["log_flux_mean"].cpu()
            tile_samples["log_flux_sd"] = dist_params_n_src["log_flux_sd"].cpu()
            tile_map_list.append(tile_samples)
    tile_samples = encoder.collate(tile_map_list)
    flat_samples = {k: v.squeeze(0) for k, v in tile_samples.items()}
    est_tile_catalog = TileCatalog.from_flat_dict(tile_slen, n_tiles_h, n_tiles_w, flat_samples)
    est_with_true_counts = est_tile_catalog.cpu().to_full_params()

    # collect into final tensors that have a single dimension
    tfluxes = []
    efluxes = []
    tlfluxes = []
    elfluxes = []
    sd_elfluxes = []

    for ii in range(batch_size):
        n_sources_ii = truth_at_most_one.n_sources[ii].item()
        assert n_sources_ii == est_with_true_counts.n_sources[ii].item()
        for jj in range(n_sources_ii):
            tlflux = truth_at_most_one["star_log_fluxes"][ii][jj].item()
            elflux = est_with_true_counts["star_log_fluxes"][ii][jj].item()
            sd_elflux = est_with_true_counts["log_flux_sd"][ii][jj].item()
            assert tlflux != 0 and elflux != 0
            tflux = math.exp(tlflux)
            eflux = math.exp(elflux)
            tfluxes.append(tflux)
            efluxes.append(eflux)
            tlfluxes.append(tlflux)
            elfluxes.append(elflux)
            sd_elfluxes.append(sd_elflux)

    return {
        "true_fluxes": torch.tensor(tfluxes),
        "est_fluxes": torch.tensor(efluxes),
        "true_log_fluxes": torch.tensor(tlfluxes),
        "est_log_fluxes": torch.tensor(elfluxes),
        "est_sd_log_fluxes": torch.tensor(sd_elfluxes),
    }


def _report_posterior_flux_calibration(model_names: list, data: dict):
    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    cis = np.linspace(0.05, 1, 20)
    cis[-1] = 0.99
    sigmas = stats.norm.interval(cis)[1]

    for ii, mname in enumerate(model_names):
        matched_fluxes = data[mname]["matched_fluxes"]
        tlfluxes = matched_fluxes["true_log_fluxes"].reshape(-1)
        elfluxes = matched_fluxes["est_log_fluxes"].reshape(-1)
        sd_elfluxes = matched_fluxes["est_sd_log_fluxes"].reshape(-1)

        fractions = []
        for s in sigmas:
            counts = tlfluxes < elfluxes + s * sd_elfluxes
            counts &= tlfluxes > elfluxes - s * sd_elfluxes
            fractions.append(counts.float().mean().item())
        fractions = np.array(fractions)

        ax.plot(cis, fractions, "-x", color=CB_color_cycle[ii], label=latex_names[mname])

    tick_labels = [0, 0.25, 0.5, 0.75, 1.0]
    ax.plot(cis, cis, "--k", label="calibrated")
    ax.legend(loc="best", prop={"size": 16})
    ax.set_xlabel(r"\rm Target coverage", fontsize=24)
    ax.set_ylabel(r"\rm Realized coverage", fontsize=24)
    ax.set_xticks(tick_labels)
    ax.set_yticks(tick_labels)
    return fig


def _stack_data(data: dict, model_names: List[str], output_names: List[str], seeds: List[int]):
    """Stack data so that there is a single tensor for all seeds."""
    new_data = {m: {output_name: {} for output_name in output_names} for m in model_names}
    for mname in model_names:
        for oname in output_names:
            tensor_names = data[mname][seeds[-1]][oname]
            new_output = {k: torch.tensor([]) for k in tensor_names}
            for seed in seeds:
                for k, t in data[mname][seed][oname].items():
                    t = t.reshape(1, *t.shape)
                    new_output[k] = torch.concat([new_output[k], t], axis=0)
            new_data[mname][oname] = new_output
    return new_data


def _create_money_plot(mag_bins: Tensor, model_names: List[str], data: dict):

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 7))

    x = (mag_bins[:, 1] + mag_bins[:, 0]) / 2

    for ii, mname in enumerate(model_names):
        color = CB_color_cycle[ii]

        # collect data from model
        precision = data[mname]["mag_bin_metrics"]["precision"]
        boot_precision = data[mname]["boot_mag_bin_metrics"]["precision"]
        recall = data[mname]["mag_bin_metrics"]["recall"]
        boot_recall = data[mname]["boot_mag_bin_metrics"]["recall"]
        matched_fluxes = data[mname]["matched_fluxes"]
        tfluxes = matched_fluxes["true_fluxes"].reshape(-1)
        efluxes = matched_fluxes["est_fluxes"].reshape(-1)

        # flatten (across seeds)
        precision = precision.mean(axis=0)
        boot_precision = boot_precision.reshape(-1, boot_precision.shape[-1])
        recall = recall.mean(axis=0)
        boot_recall = boot_recall.reshape(-1, boot_recall.shape[-1])

        # precision
        precision1 = boot_precision.quantile(0.25, 0)
        precision2 = boot_precision.quantile(0.75, 0)
        ax1.plot(x, precision, "-o", color=color, label=latex_names[mname], markersize=6)
        ax1.fill_between(x, precision1, precision2, color=color, alpha=0.5)

        # recall
        recall1 = boot_recall.quantile(0.25, 0)
        recall2 = boot_recall.quantile(0.75, 0)
        ax2.plot(x, recall, "-o", color=color, label=latex_names[mname], markersize=6)
        ax2.fill_between(x, recall1, recall2, color=color, alpha=0.5)

        # fluxes
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
    seeds = cfg.results.seeds
    output_names = ["mag_bin_metrics", "boot_mag_bin_metrics", "matched_fluxes"]
    background_obj = instantiate(cfg.datasets.galsim_blends.background)

    if cfg.results.overwrite:
        outputs = {model_name: {} for model_name in model_names}

        for seed in seeds:
            for model_name in model_names:
                # run model and get catalogs
                model_path = f"{cfg.results.models_path}/{model_name}_encoder_{seed}.pt"
                test_images = all_test_images[model_name]
                background = background_obj.sample(test_images.shape)
                encoder = _load_encoder(cfg, model_path)
                est = _run_model_on_images(encoder, test_images, background)

                # need galaxy_bools key to be present, even if only stars.
                est["galaxy_bools"] = torch.zeros(est.batch_size, est.max_sources, 1)

                # add mags to catalog
                truth = _add_mags_to_catalog(truth)
                est = _add_mags_to_catalog(est)

                # get all metrics from catalogs
                bin_metrics = _compute_mag_bin_metrics(mag_bins, truth, est)
                boot_metrics = _get_bootstrap_pr_err(10000, mag_bins, truth, est)
                matched_fluxes = _get_fluxes_cond_true_detections(
                    encoder, truth, test_images, background
                )

                outputs[model_name][seed] = {
                    "mag_bin_metrics": bin_metrics,
                    "boot_mag_bin_metrics": boot_metrics,
                    "matched_fluxes": matched_fluxes,
                }
                assert list(outputs[model_name][seed].keys()) == output_names

        torch.save(outputs, cfg.results.cache_results)

    else:
        assert Path(cfg.results.cache_results).exists()

        fig_path = Path(cfg.results.figs_path)
        fig_path.mkdir(exist_ok=True)

        # stack data with different seeds
        raw_data = torch.load(cfg.results.cache_results)
        data = _stack_data(raw_data, model_names, output_names, seeds)

        # create all figures
        figs = []
        figs.append(_create_money_plot(mag_bins, model_names, data))
        figs.append(_report_posterior_flux_calibration(model_names, data))

        for fig, name in zip(figs, cfg.results.figs):
            fig.savefig(fig_path / f"{name}.png", format="png")


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
