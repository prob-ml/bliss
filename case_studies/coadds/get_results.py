#!/usr/bin/env python3
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import hydra
import torch
from hydra.utils import instantiate
from matplotlib import pyplot as plt
from torch import Tensor

from bliss.catalog import FullCatalog
from bliss.datasets.sdss import convert_flux_to_mag
from bliss.encoder import Encoder
from bliss.reporting import DetectionMetrics
from case_studies.sdss_galaxies.plots.bliss_figures import CB_color_cycle

device = torch.device("cuda:0")

latex_names = {
    "single": r"\rm single exposure",
    "coadd_10": r"\rm Coadd $d=10$",
    "coadd_25": r"\rm Coadd $d=25$",
    "coadd_50": r"\rm Coadd $d=50$",
}


def _compute_mag_bin_metrics(
    mag_bins: Tensor, truth: FullCatalog, est: FullCatalog
) -> Dict[str, Tensor]:
    metrics_per_mag = defaultdict(lambda: torch.zeros(len(mag_bins)))

    # compute data for precision/recall/classification accuracy as a function of magnitude.
    for ii, (mag1, mag2) in enumerate(mag_bins):
        detection_metrics = DetectionMetrics()

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


def load_test_dataset(path) -> Tuple[dict, FullCatalog]:
    test_ds = torch.load(path)
    image_keys = {"coadd_10", "coadd_25", "coadd_50", "single"}
    all_keys = list(test_ds.keys())
    truth_params = {k: test_ds.pop(k) for k in all_keys if k not in image_keys}
    truth_params["n_sources"] = truth_params["n_sources"].reshape(-1)
    truth_cat = FullCatalog(88, 88, truth_params)
    return test_ds, truth_cat


def load_encoder(cfg, model_path) -> Encoder:
    model = instantiate(cfg.models.detection_encoder).to(device).eval()
    model.load_state_dict(torch.load(model_path, map_location=device))
    return Encoder(model.eval(), n_images_per_batch=10, n_rows_per_batch=15).to(device).eval()


def running_model_on_images(encoder, images, background) -> FullCatalog:
    tile_est = encoder.variational_mode(images, background)
    return tile_est.cpu().to_full_params()


def add_mags_to_catalog(cat: FullCatalog) -> FullCatalog:
    batch_size = cat.n_sources.shape[0]
    cat["mags"] = torch.zeros_like(cat["star_fluxes"])
    for ii in range(batch_size):
        n_sources = cat.n_sources[ii].item()
        cat["mags"][ii, :n_sources] = convert_flux_to_mag(cat["star_fluxes"][ii, :n_sources])
    return cat


def bootstrap_fn(
    n_samples: int,
    fn: Callable,
    truth: FullCatalog,
    est: FullCatalog,
    *fn_args,
) -> List:
    h, w = truth.height, truth.width

    truth_dict = {**truth}
    truth_dict["n_sources"] = truth.n_sources
    truth_dict["plocs"] = truth.plocs

    est_dict = {**est}
    est_dict["n_sources"] = truth.n_sources
    est_dict["plocs"] = truth.plocs

    metrics = []
    for _ in range(n_samples):
        boot_truth_dict = {}
        boot_est_dict = {}
        boostrap_indices = torch.randint(0, len(truth), (len(truth),))
        for k in truth_dict:  # noqa: WPS528
            boot_truth_dict[k] = truth_dict[k][boostrap_indices]
            boot_est_dict[k] = est_dict[k][boostrap_indices]
        boot_truth_cat = FullCatalog(h, w, boot_truth_dict)
        boot_est_cat = FullCatalog(h, w, boot_est_dict)
        metric = fn(*fn_args, boot_truth_cat, boot_est_cat)
        metrics.append(metric)

    return metrics


def create_money_plot(mag_bins: Tensor, model_names: List[str], data: dict):

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 7))

    for ii, mname in enumerate(model_names):
        color = CB_color_cycle[ii]
        precision = data[mname]["mag_bin_metrics"]["precision"]
        recall = data[mname]["mag_bin_metrics"]["recall"]
        ax1.plot(mag_bins[:, 1], precision, "-o", label=latex_names[mname], color=color)
        ax2.plot(mag_bins[:, 1], recall, "-o", label=latex_names[mname], color=color)

    ax1.xlabel(r"\rm Magnitude")
    ax2.xlabel(r"\rm Magnitude")
    ax1.ylabel(r"\rm Precision")
    ax2.ylabel(r"\rm Recall")
    ax1.legend(loc="best", prop={"size": 14})
    ax2.legend(loc="best", prop={"size": 14})
    ax1.minorticks_on()
    ax2.minorticks_on()

    return fig


@hydra.main(config_path="./config", config_name="config")
def main(cfg):

    mag_cuts2 = torch.arange(17.0, 23.5, 0.5)
    mag_cuts1 = torch.full_like(mag_cuts2, fill_value=-torch.inf)
    mag_bins = torch.column_stack((mag_cuts1, mag_cuts2))

    model_names = cfg.evaluation.models  # e.g. 'coadd50', 'single'

    if cfg.evaluation.overwrite:
        test_path = cfg.evaluation.test_path
        all_test_images, truth = load_test_dataset(test_path)
        background_obj = instantiate(cfg.datasets.galsim_blends.background)

        outputs = {}

        for model_name in model_names:
            model_path = f"output/{model_name}_encoder.pt"
            test_images = all_test_images[model_name]
            background = background_obj.sample(test_images.shape)
            encoder = load_encoder(cfg, model_path)
            est = running_model_on_images(encoder, test_images, background)
            est["galaxy_bools"] = torch.zeros(est.batch_size, est.max_sources, 1)

            truth = add_mags_to_catalog(truth)
            est = add_mags_to_catalog(est)

            bin_metrics = _compute_mag_bin_metrics(mag_bins, truth, est)

            outputs[model_name] = {"mag_bin_metrics": bin_metrics}

        torch.save(outputs, "output/metric_results.pt")

    else:
        fig_path = Path("output/figs")
        fig_path.mkdir(exist_ok=True)

        data = torch.load("output/metrics_results.pt")

        # create all figures
        figs = []
        figs.append(create_money_plot(mag_bins, model_names, data))

        for fig in figs:
            fig.savefig(fig_path / "money.png", format="png")


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
