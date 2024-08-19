# flake8: noqa
# pylint: skip-file
# Ignoring flake8/pylint for this file since this is just a plotting script

import argparse
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from hydra import compose, initialize
from hydra.utils import instantiate
from sklearn.metrics import roc_auc_score, roc_curve
from torch.utils.data import DataLoader
from tqdm import tqdm

from bliss.catalog import TileCatalog, convert_mag_to_nmgy

# Set up plots, colors, global constants
sns.set_theme("paper")
matplotlib.rc("text", usetex=True)
plt.rc("font", family="serif")

COLORS = [
    "#0072BD",  # blue
    "#D95319",  # orange
    "#EDB120",  # yellow
    "#7E2F8E",  # purple
    "#77AC30",  # green
    "#4DBEEE",  # light blue
    "#A2142F",  # dark red
]

BRIGHT_THRESHOLD = 17.777
FAINT_THRESHOLD = 21.746

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--run_eval", action="store_true")
parser.add_argument("--plot_eval", action="store_true")
parser.add_argument("--run_calibration", action="store_true")
parser.add_argument("--plot_calibration", action="store_true")
parser.add_argument("--data_path", type=str, required=True, help="Path to test data directory")

args = parser.parse_args()

# Load config, data, models
with initialize(config_path="./conf", version_base=None):
    base_cfg = compose("config")

data_path = args.data_path  # "/data/scratch/aakash/multi_field"
dataset_name = data_path.split("/")[-1]  # used for save directory names

cached_dataset = instantiate(
    base_cfg.cached_simulator, cached_data_path=data_path, splits="0:0/0:0/90:100"
)
cached_dataset.setup(stage="test")
calib_dataloader = cached_dataset.test_dataloader()
print(f"Test dataset size: {len(cached_dataset.test_dataset)}")

trainer = instantiate(base_cfg.train.trainer, logger=None)

models = {
    "single_field": {
        "ckpt_path": "/home/aakashdp/bliss_output/PSF_MODELS/single_field_with_gal_params/checkpoints/best_encoder.ckpt",
        "config_path": "single_field.yaml",
        "plot_config": {"name": "Single-field", "marker": "o", "color": COLORS[0]},
    },
    "psf_unaware": {
        "ckpt_path": "/home/aakashdp/bliss_output/PSF_MODELS/psf_unaware_with_gal_params/checkpoints/best_encoder.ckpt",
        "config_path": "psf_unaware.yaml",
        "plot_config": {"name": "PSF-unaware", "marker": "s", "color": COLORS[1]},
    },
    "psf_aware": {
        "ckpt_path": "/home/aakashdp/bliss_output/PSF_MODELS/psf_aware_with_gal_params/checkpoints/best_encoder.ckpt",
        "config_path": "psf_aware.yaml",
        "plot_config": {"name": "PSF-aware", "marker": "^", "color": COLORS[2]},
    },
}

rep_key = list(models.keys())[0]

for model_name, model_info in models.items():
    with initialize(config_path="./conf", version_base=None):
        cfg = compose(model_info["config_path"])

    encoder = instantiate(cfg.encoder)
    encoder.load_state_dict(torch.load(model_info["ckpt_path"], map_location="cpu")["state_dict"])
    encoder.eval()
    model_info["encoder"] = encoder
    model_info["config"] = cfg


def run_eval():
    """Compute metrics and standard deviations for each model."""
    # Compute metrics for each model
    for model_name in models:
        print(f"Evaluating {model_name} model...")
        results = trainer.test(
            models[model_name]["encoder"], datamodule=cached_dataset, verbose=False
        )
        models[model_name]["results"] = results

    # Compute bootstrap variance
    N_samples = 5
    orig_test_slice = cached_dataset.slices[2]
    orig_start = orig_test_slice.start
    orig_stop = orig_test_slice.stop

    data_for_var = {
        model: {key: [] for key in models[rep_key]["results"][0].keys()} for model in models
    }

    for i in tqdm(range(N_samples), desc=f"Bootstrapping {N_samples} samples"):
        random_batch = np.random.randint(orig_start, orig_stop - 1)
        cached_dataset.slices[2] = slice(random_batch, random_batch + 1)
        cached_dataset.setup(stage="test")

        for model_name in models:
            results = trainer.test(
                models[model_name]["encoder"], dataloaders=cached_dataset, verbose=False
            )

            for key, val in results[0].items():
                data_for_var[model_name][key].append(val)

    cached_dataset.slices[2] = orig_test_slice

    stds = {
        model: {f"{key}_std": np.nanstd(val) for key, val in data_for_var[model].items()}
        for model in data_for_var
    }

    # Concatenate results into dataframe
    keys = list(models[rep_key]["results"][0].keys())
    keys.extend(stds[rep_key].keys())

    data = {}
    for model_name in models:
        results = models[model_name]["results"][0] | stds[model_name]
        model_vals = [results[key] for key in keys]
        data[model_name] = model_vals

    data_flat = pd.DataFrame.from_dict(
        data, orient="index", columns=[key.split("/")[-1] for key in keys]
    ).reset_index()
    data_flat = data_flat.rename(columns={"index": "model"})
    data_flat = data_flat.set_index("model")

    os.makedirs(f"data/{dataset_name}", exist_ok=True)
    with open(f"data/{dataset_name}/metrics.pt", "wb") as f:
        torch.save(data_flat.to_dict(), f)


def run_calibration():
    """Get posterior distributions for each model and save to disk."""
    # Precompute predicted distributions
    pred_dists = {}
    assert torch.cuda.is_available(), "ERROR: GPU not found."
    device = "cuda"

    with torch.no_grad():
        for model_name, model in models.items():
            encoder = model["encoder"].to(device)
            pred_dists[model_name] = []

            for batch in tqdm(
                calib_dataloader, desc=f"Getting calibration metrics for {model_name}..."
            ):
                batch_size, _n_bands, h, w = batch["images"].shape[0:4]
                ht, wt = h // encoder.tile_slen, w // encoder.tile_slen

                target_cat = TileCatalog(batch["tile_catalog"])
                target_cat = target_cat.filter_by_flux(min_flux=1.59, band=2)
                target_cat = target_cat.to(device)

                batch["images"] = batch["images"].to(device)
                batch["psf_params"] = batch["psf_params"].to(device)

                # Get predicted params
                x_features = encoder.get_features(batch)
                patterns_to_use = (0,)  # no checkerboard
                mask_pattern = encoder.mask_patterns[patterns_to_use, ...][0]
                mask = mask_pattern.repeat([batch_size, ht // 2, wt // 2])
                context1 = encoder.make_context(target_cat, mask)
                x_cat1 = encoder.catalog_net(x_features, context1)
                factor_param_zip = encoder.var_dist._factor_param_pairs(x_cat1)

                batch_dists = {}
                for factor, params in factor_param_zip:
                    params = params.to("cpu")
                    factor_name = factor.name
                    dist = factor._get_dist(params)
                    batch_dists[factor_name] = dist

                pred_dists[model_name].append(batch_dists)

    os.makedirs("data", exist_ok=True)
    torch.save(pred_dists, f"data/{dataset_name}/posterior_dists.pt")


def plot_metric(data, metric, metric_cfg):
    """Plot results for a specific metric.

    Args:
        data (dict): All metrics results computed by run_eval.
        metric (str): Name of metric to plot. Should be a key in data.
        metric_cfg (dict): Config info like the type of metric and ylabel to use for the plot.
    """
    fig, ax = plt.subplots(figsize=(7.25, 5))

    bins = [17.777, 19.101, 19.781, 20.258, 20.625, 20.940, 21.227, 21.495, 21.746, 22.000]
    n_bins = len(bins)
    xlabel = "r-band magnitude"

    # Plot each model
    for i, name in enumerate(data[metric].keys()):
        plot_config = models[name]["plot_config"]

        # Get metric values in each bin
        binned_vals = np.array([data[f"{metric}_bin_{j}"][name] for j in range(n_bins)])

        # Construct label for legend: "model_name [average value for metric]""
        model_name = plot_config["name"]
        average_val = data[metric][name]
        label = f"{model_name} [{average_val:.3f}]"

        # Plot line
        ax.plot(
            binned_vals,
            c=plot_config["color"],
            markeredgecolor="k",
            markersize=6,
            linewidth=1,
            marker=plot_config["marker"],
            label=label,
        )

        # Fill in +/- one standard deviation
        if f"{metric}_bin_0_std" in data and name in data[f"{metric}_bin_0_std"]:
            binned_stds = np.array([data[f"{metric}_bin_{j}_std"][name] for j in range(n_bins)])
            lower = binned_vals - binned_stds
            upper = binned_vals + binned_stds

            ax.fill_between(
                np.arange(len(lower)), lower, upper, color=plot_config["color"], alpha=0.2
            )

    # Place bin values on xticks
    xticklabels = [f"{bins[i+1]:.2f}" for i in range(n_bins - 1)]
    xticklabels.insert(0, f"$<${bins[0]:.2f}")
    ax.set_xticks(range(len(xticklabels)), xticklabels)
    ax.tick_params(axis="both", which="major", labelsize="x-large")

    if metric_cfg["yaxis_in_percent"]:
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1.0))

    # Set axis labels and legend
    ax.set_xlabel(xlabel, fontsize="xx-large")
    ax.set_ylabel(metric_cfg["ylabel"], fontsize="xx-large")
    ax.legend(fontsize="x-large")

    # Save figure
    fig.tight_layout()
    plt.savefig(f"plots/{dataset_name}/metrics/{metric}.pdf")


def compute_expected_sources(pred_dists, bins, cached_path):
    """Compute expected value of number of sources and save to disk.

    Args:
        pred_dists (dict): Dictionary of calibration results computed by run_calibration.
        bins (list): List of magnitude bins.
        cached_path (str): Where to save the resulting data.

    Returns:
        dict: Dictionary of results.
    """
    n_bins = len(bins)
    sum_all = {name: torch.zeros(n_bins) for name in models}
    all_count = {name: torch.zeros(n_bins) for name in models}

    sum_bright = {name: torch.zeros(n_bins) for name in models}
    bright_count = {name: torch.zeros(n_bins) for name in models}

    sum_dim = {name: torch.zeros(n_bins) for name in models}
    dim_count = {name: torch.zeros(n_bins) for name in models}

    for i, batch in enumerate(tqdm(calib_dataloader, desc="Computing expected number of sources")):
        target_cat = TileCatalog(batch["tile_catalog"])
        target_cat = target_cat.filter_by_flux(min_flux=1.59, band=2)

        normal_mask = (target_cat.on_magnitudes(c=1)[..., 2] < 22).squeeze()
        bright_mask = (target_cat.on_magnitudes(c=1)[..., 2] < BRIGHT_THRESHOLD).squeeze()
        dim_mask = (target_cat.on_magnitudes(c=1)[..., 2] > FAINT_THRESHOLD).squeeze() * normal_mask

        true_sources = (target_cat["n_sources"].bool() * normal_mask).sum(dim=(1, 2))
        true_bright = (target_cat["n_sources"].bool() * bright_mask).sum(dim=(1, 2))
        true_dim = (target_cat["n_sources"].bool() * dim_mask).sum(dim=(1, 2))

        binned_true = torch.bucketize(true_sources, bins)
        binned_bright = torch.bucketize(true_bright, bins)
        binned_dim = torch.bucketize(true_dim, bins)

        for name in models:
            on_prob = pred_dists[name][i]["n_sources"].probs[..., 1]

            all_on = on_prob.sum(dim=(1, 2))
            bright_on = (on_prob * bright_mask).sum(dim=(1, 2))
            dim_on = (on_prob * dim_mask).sum(dim=(1, 2))

            tmp = torch.zeros(n_bins, dtype=on_prob.dtype)
            sum_all[name] += tmp.scatter_add(0, binned_true, all_on)
            all_count[name] += binned_true.bincount(minlength=n_bins)

            tmp = torch.zeros(n_bins, dtype=on_prob.dtype)
            sum_bright[name] += tmp.scatter_add(0, binned_bright, bright_on)
            bright_count[name] += binned_bright.bincount(minlength=n_bins)

            tmp = torch.zeros(n_bins, dtype=on_prob.dtype)
            sum_dim[name] += tmp.scatter_add(0, binned_dim, dim_on)
            dim_count[name] += binned_dim.bincount(minlength=n_bins)

    source_data = {"mean_sources_per_bin": {}, "mean_bright_per_bin": {}, "mean_dim_per_bin": {}}
    for name in models:
        source_data["mean_sources_per_bin"][name] = sum_all[name] / all_count[name]
        source_data["mean_bright_per_bin"][name] = sum_bright[name] / bright_count[name]
        source_data["mean_dim_per_bin"][name] = sum_dim[name] / dim_count[name]

    torch.save(source_data, cached_path)
    return source_data


def plot_expected_vs_predicted_sources(mean_sources_per_bin, mean_bright_per_bin, mean_dim_per_bin):
    """Plot the expected value of number of sources vs true number of sources.

    Args:
        mean_sources_per_bin (dict): Average number of sources in each bin.
        mean_bright_per_bin (dict): Average number of bright sources in each bin.
        mean_dim_per_bin (dict): Average number of dim sources in each bin.
    """
    fig, ax = plt.subplots(1, 3, figsize=(5 * 3, 5))

    bins = torch.arange(20)
    n_bins = len(bins)
    shared_params = {"markeredgecolor": "k", "markersize": 6, "linewidth": 1}

    # All
    max_n_all = max(
        [
            torch.argmax(torch.arange(n_bins) * ~mean_sources_per_bin[name].isnan())
            for name in models
        ]
    )
    ax[0].plot(torch.arange(max_n_all + 1), c="darkgray", linewidth=1, linestyle="dashed")
    for name in mean_sources_per_bin:
        plot_config = models[name]["plot_config"]
        ax[0].plot(
            mean_sources_per_bin[name],
            c=plot_config["color"],
            marker=plot_config["marker"],
            label=plot_config["name"],
            **shared_params,
        )
    ax[0].legend(fontsize="x-large")
    ax[0].set_title("All sources", fontsize="xx-large")

    # Bright
    max_n_bright = max(
        [torch.argmax(torch.arange(n_bins) * ~mean_bright_per_bin[name].isnan()) for name in models]
    )
    ax[1].plot(torch.arange(max_n_bright + 1), c="darkgray", linewidth=1, linestyle="dashed")
    for name in mean_bright_per_bin:
        plot_config = models[name]["plot_config"]
        ax[1].plot(
            mean_bright_per_bin[name],
            c=plot_config["color"],
            marker=plot_config["marker"],
            label=plot_config["name"],
            **shared_params,
        )
    ax[1].set_title(f"Bright (magnitude $<$ {BRIGHT_THRESHOLD:.2f})", fontsize="xx-large")

    # Faint
    max_n_dim = max(
        [torch.argmax(torch.arange(n_bins) * ~mean_dim_per_bin[name].isnan()) for name in models]
    )
    ax[2].plot(torch.arange(max_n_dim + 1), c="darkgray", linewidth=1, linestyle="dashed")
    for name in mean_dim_per_bin:
        plot_config = models[name]["plot_config"]
        ax[2].plot(
            mean_dim_per_bin[name],
            c=plot_config["color"],
            marker=plot_config["marker"],
            label=plot_config["name"],
            **shared_params,
        )
    ax[2].set_title(f"Faint (magnitude {FAINT_THRESHOLD:.2f}-22)", fontsize="xx-large")

    for a in ax:
        a.tick_params(axis="both", which="major", labelsize="x-large")
        a.set_xlabel("True sources", fontsize="xx-large")
    ax[0].set_ylabel("Detected sources", fontsize="xx-large")

    fig.tight_layout()
    plt.savefig(f"plots/{dataset_name}/calibration/true_vs_pred_sources_by_mag.pdf")


def compute_prob_flux_within_one_mag(pred_dists, bins, cached_path):
    """Compute probability of estimated flux being within 1 magnitude of true flux.

    Args:
        pred_dists (dict): Calibration results computed by run_calibration
        bins (list): Magnitude bins
        cached_path (str): Where to save the results

    Returns:
        Dict: dictionary of results
    """
    n_bins = len(bins)
    sum_probs = {name: torch.zeros(n_bins) for name in models}
    bin_count = {name: torch.zeros(n_bins) for name in models}

    for i, batch in enumerate(tqdm(calib_dataloader, desc="Prob flux within 1 of true mag")):
        # Get target catalog and magnitudes, construct upper and lower bounds
        target_cat = TileCatalog(batch["tile_catalog"])
        target_cat = target_cat.filter_by_flux(min_flux=1.59, band=2)

        target_mags = target_cat.on_magnitudes(c=1)
        lb = convert_mag_to_nmgy(target_mags + 1).squeeze()
        ub = convert_mag_to_nmgy(target_mags - 1).squeeze()

        target_on_mags = target_mags[target_cat.is_on_mask][:, 2].contiguous()
        binned_target_on_mags = torch.bucketize(target_on_mags, bins)

        for name, model in models.items():
            # Get probabilities of flux within 1 magnitude of true
            q_star_flux = pred_dists[name][i]["star_fluxes"].base_dist
            q_gal_flux = pred_dists[name][i]["galaxy_fluxes"].base_dist

            star_flux_probs = q_star_flux.cdf(ub) - q_star_flux.cdf(lb)
            gal_flux_probs = q_gal_flux.cdf(ub) - q_gal_flux.cdf(lb)

            pred_probs = torch.where(
                target_cat.star_bools, star_flux_probs.unsqueeze(-2), gal_flux_probs.unsqueeze(-2)
            )
            pred_probs = pred_probs[target_cat.is_on_mask][:, 2]

            probs_per_bin = torch.zeros(n_bins, dtype=pred_probs.dtype)
            sum_probs[name] += probs_per_bin.scatter_add(0, binned_target_on_mags, pred_probs)
            bin_count[name] += binned_target_on_mags.bincount(minlength=n_bins)

    binned_avg_flux_probs = {}
    for name in models:
        binned_avg_flux_probs[name] = sum_probs[name] / bin_count[name]
    torch.save(binned_avg_flux_probs, cached_path)
    return binned_avg_flux_probs


def plot_flux_within_one_mag(binned_avg_flux_probs, bins):
    """Plot the probability of estimated flux being within 1 magnitude of true flux.

    Args:
        binned_avg_flux_probs (dict): results from compute_avg_flux_probs
        bins (list): Magnitude bins
    """
    fig, ax = plt.subplots(figsize=(7.25, 5))
    for i, name in enumerate(models):
        if name not in binned_avg_flux_probs:
            continue

        plot_config = models[name]["plot_config"]

        binned_vals = binned_avg_flux_probs[name].detach()
        label = plot_config["name"]
        ax.plot(
            binned_vals,
            c=plot_config["color"],
            markeredgecolor="k",
            markersize=6,
            linewidth=1,
            marker=plot_config["marker"],
            label=label,
        )

    xticklabels = [f"{bins[i+1]:.2f}" for i in range(len(bins) - 1)]
    xticklabels.insert(0, f"$<${bins[0]:.2f}")
    ax.set_xticks(range(len(xticklabels)), xticklabels)

    ax.tick_params(axis="both", which="major", labelsize="xx-large")
    ax.set_xlabel("r-band Magnitude", fontsize="xx-large")
    ax.set_ylabel("Pr($f_{\mathrm{pred}}$ within 1 magnitude)", fontsize="xx-large")
    ax.legend(fontsize="x-large")

    fig.tight_layout()
    plt.savefig(f"plots/{dataset_name}/calibration/prob_flux_within_1_mag.pdf")


def compute_prop_flux_in_interval(pred_dists, intervals, cached_path):
    """Compute proportion of sources that fall in equal-tailed credible intervals.

    Args:
        pred_dists (dict): Calibration results computed by run_calibration
        intervals (list): List of credible interval sizes
        cached_path (str): Where to save the results

    Returns:
        Dict: dictionary of results
    """
    sum_all_in_eti = {name: torch.zeros(len(intervals)) for name in models}
    sum_bright_in_eti = {name: torch.zeros(len(intervals)) for name in models}
    sum_dim_in_eti = {name: torch.zeros(len(intervals)) for name in models}
    all_count = 0
    bright_count = 0
    dim_count = 0

    for i, batch in enumerate(tqdm(calib_dataloader, desc="Computing prob in credible interval")):
        target_cat = TileCatalog(batch["tile_catalog"])
        target_cat = target_cat.filter_by_flux(min_flux=1.59, band=2)
        true_fluxes = target_cat.on_fluxes[..., 0, 2]

        normal_mask = (target_cat.on_magnitudes(c=1)[..., 2] < 22).squeeze()
        bright_mask = (target_cat.on_magnitudes(c=1)[..., 2] < BRIGHT_THRESHOLD).squeeze()
        dim_mask = (target_cat.on_magnitudes(c=1)[..., 2] > FAINT_THRESHOLD).squeeze() * normal_mask

        all_count += target_cat["n_sources"].sum()
        bright_count += bright_mask.sum()
        dim_count += dim_mask.sum()

        for name, model in models.items():
            q_star_flux = pred_dists[name][i]["star_fluxes"].base_dist
            q_gal_flux = pred_dists[name][i]["galaxy_fluxes"].base_dist

            for j, interval in enumerate(intervals):
                # construct equal tail intervals and determine if true flux is within ETI
                tail_prob = (1 - interval) / 2
                star_lb = q_star_flux.icdf(tail_prob)[..., 2]
                star_ub = q_star_flux.icdf(1 - tail_prob)[..., 2]
                gal_lb = q_gal_flux.icdf(tail_prob)[..., 2]
                gal_ub = q_gal_flux.icdf(1 - tail_prob)[..., 2]

                star_flux_in_eti = (true_fluxes >= star_lb) & (true_fluxes <= star_ub)
                gal_flux_in_eti = (true_fluxes >= gal_lb) & (true_fluxes <= gal_ub)

                source_in_eti = torch.where(
                    target_cat.star_bools.squeeze(), star_flux_in_eti, gal_flux_in_eti
                )

                sum_all_in_eti[name][j] += (source_in_eti * target_cat.is_on_mask.squeeze()).sum()
                sum_bright_in_eti[name][j] += (source_in_eti * bright_mask).sum()
                sum_dim_in_eti[name][j] += (source_in_eti * dim_mask).sum()

    # Compute proportions and save data
    prop_all_in_eti = {}
    prop_bright_in_eti = {}
    prop_dim_in_eti = {}
    for name in models:
        prop_all_in_eti[name] = sum_all_in_eti[name] / all_count
        prop_bright_in_eti[name] = sum_bright_in_eti[name] / bright_count
        prop_dim_in_eti[name] = sum_dim_in_eti[name] / dim_count

    data = {
        "prop_all_in_eti": prop_all_in_eti,
        "prop_bright_in_eti": prop_bright_in_eti,
        "prop_dim_in_eti": prop_dim_in_eti,
    }
    torch.save(data, cached_path)
    return data


def plot_prop_flux_in_interval(prop_all_in_eti, prop_bright_in_eti, prop_dim_in_eti, intervals):
    """Plot proportion of sources that fall in credible interval.

    Args:
        prop_all_in_eti (dict): From compute_prop_in_interval
        prop_bright_in_eti (dict): From compute_prop_in_interval
        prop_dim_in_eti (dict): From compute_prop_in_interval
        intervals (List): List of credible intervals
    """
    fig, ax = plt.subplots(1, 3, figsize=(5 * 3, 5))

    ax[0].plot(intervals, intervals, color="darkgray", linewidth=1, linestyle="dashed")
    ax[1].plot(intervals, intervals, color="darkgray", linewidth=1, linestyle="dashed")
    ax[2].plot(intervals, intervals, color="darkgray", linewidth=1, linestyle="dashed")

    for name in models:
        plot_config = models[name]["plot_config"]
        kwargs = {
            "color": plot_config["color"],
            "marker": plot_config["marker"],
            "markeredgecolor": "k",
            "markersize": 6,
            "linewidth": 1,
            "label": plot_config["name"],
        }
        ax[0].plot(intervals, prop_all_in_eti[name], **kwargs)
        ax[1].plot(intervals, prop_bright_in_eti[name], **kwargs)
        ax[2].plot(intervals, prop_dim_in_eti[name], **kwargs)

    ax[0].legend(fontsize="x-large")
    ax[0].set_title("All sources", fontsize="xx-large")
    ax[1].set_title(f"Bright (magnitude $<$ {BRIGHT_THRESHOLD:.2f})", fontsize="xx-large")
    ax[2].set_title(f"Faint (magnitude {FAINT_THRESHOLD:.2f}-22)", fontsize="xx-large")

    xticks = (intervals * 100).int().tolist()
    for a in ax:
        a.set_xticks(intervals, xticks)
        a.tick_params(axis="both", which="major", labelsize="x-large")
        a.set_xlabel("\% Credible Interval", fontsize="xx-large")
    ax[0].set_ylabel("Proportion of true fluxes in interval", fontsize="xx-large")

    fig.tight_layout()
    plt.savefig(f"plots/{dataset_name}/calibration/prop_flux_in_interval_by_mag.pdf")


def compute_avg_prob_true_source_type(pred_dists, bins, cached_path):
    """Compute average probability of correct class by magnitude bin.

    Args:
        pred_dists (dict): Calibration results computed by run_calibration
        bins (list): Magnitude bins
        cached_path (str): Where to save the results

    Returns:
        Dict: dictionary of results
    """
    n_bins = len(bins)
    sum_probs = {name: torch.zeros(n_bins) for name in models}
    bin_count = {name: torch.zeros(n_bins) for name in models}

    for i, batch in enumerate(tqdm(calib_dataloader, desc="Computing prob of true class")):
        target_cat = TileCatalog(batch["tile_catalog"])
        target_cat = target_cat.filter_by_flux(min_flux=1.59, band=2)

        target_mags = target_cat.on_magnitudes(c=1)
        target_on_mags = target_mags[target_cat.is_on_mask][:, 2].contiguous()
        binned_target_on_mags = torch.bucketize(target_on_mags, bins)

        target_types = target_cat["source_type"].squeeze().bool()

        for name, model in models.items():
            gal_probs = pred_dists[name][i]["source_type"].probs[..., 1]

            true_type_prob = torch.where(target_types, gal_probs, 1 - gal_probs)
            on_gal_probs = true_type_prob[target_cat.is_on_mask.squeeze()]

            probs_per_bin = torch.zeros(n_bins, dtype=true_type_prob.dtype)
            sum_probs[name] += probs_per_bin.scatter_add(0, binned_target_on_mags, on_gal_probs)
            bin_count[name] += binned_target_on_mags.bincount(minlength=n_bins)

    binned_source_type_probs = {}
    for name in models:
        binned_source_type_probs[name] = sum_probs[name] / bin_count[name]
    torch.save(binned_source_type_probs, cached_path)
    return binned_source_type_probs


def plot_prob_true_source_type(binned_source_type_probs, bins):
    """Plot average probability of true source type.

    Args:
        binned_source_type_probs (dict): From compute_source_type_probs
        bins (list): Magnitude bins
    """
    fig, ax = plt.subplots(figsize=(7.25, 5))

    for name in binned_source_type_probs:
        plot_config = models[name]["plot_config"]

        binned_vals = binned_source_type_probs[name].detach()
        ax.plot(
            binned_vals,
            c=plot_config["color"],
            markeredgecolor="k",
            markersize=6,
            linewidth=1,
            marker=plot_config["marker"],
            label=plot_config["name"],
        )

    xticklabels = [f"{bins[i+1]:.2f}" for i in range(len(bins) - 1)]
    xticklabels.insert(0, f"$<${bins[0]:.2f}")

    ax.set_xticks(range(len(xticklabels)), xticklabels)
    ax.tick_params(axis="both", which="major", labelsize="xx-large")
    ax.set_xlabel("r-band Magnitude", fontsize="xx-large")
    ax.set_ylabel("Probability of correct classification", fontsize="xx-large")
    ax.legend(fontsize="x-large")

    fig.tight_layout()
    plt.savefig(f"plots/{dataset_name}/calibration/prob_true_source_type.pdf")


def compute_classification_probs_by_threshold(pred_dists, thresholds, cached_path):
    """Compute probability of correct classification by decision threshold.

    Args:
        pred_dists (dict): Calibration results computed by run_calibration
        thresholds (list): Classification thresholds
        cached_path (str): Where to save the results

    Returns:
        Dict: dictionary of results
    """
    pred_all_gal = {name: torch.zeros(len(thresholds)) for name in models}
    pred_bright_gal = {name: torch.zeros(len(thresholds)) for name in models}
    pred_dim_gal = {name: torch.zeros(len(thresholds)) for name in models}

    pred_all_star = {name: torch.zeros(len(thresholds)) for name in models}
    pred_bright_star = {name: torch.zeros(len(thresholds)) for name in models}
    pred_dim_star = {name: torch.zeros(len(thresholds)) for name in models}

    true_all_gal = 0
    true_bright_gal = 0
    true_dim_gal = 0

    true_all_star = 0
    true_bright_star = 0
    true_dim_star = 0

    for i, batch in enumerate(tqdm(calib_dataloader, desc="Prob correct star/gal by threshold")):
        target_cat = TileCatalog(batch["tile_catalog"])
        target_cat = target_cat.filter_by_flux(min_flux=1.59, band=2)

        normal_mask = (target_cat.on_magnitudes(c=1)[..., 2] < 22).squeeze()
        bright_mask = (target_cat.on_magnitudes(c=1)[..., 2] < BRIGHT_THRESHOLD).squeeze()
        dim_mask = (target_cat.on_magnitudes(c=1)[..., 2] > FAINT_THRESHOLD).squeeze() * normal_mask

        true_all_gal += target_cat.galaxy_bools.sum()
        true_bright_gal += (target_cat.galaxy_bools.squeeze() * bright_mask).sum()
        true_dim_gal += (target_cat.galaxy_bools.squeeze() * dim_mask).sum()

        true_all_star += target_cat.star_bools.sum()
        true_bright_star += (target_cat.star_bools.squeeze() * bright_mask).sum()
        true_dim_star += (target_cat.star_bools.squeeze() * dim_mask).sum()

        for name, model in models.items():
            gal_probs = pred_dists[name][i]["source_type"].probs[..., 1]
            star_probs = 1 - gal_probs

            all_gal_probs = gal_probs * target_cat.galaxy_bools.squeeze()
            bright_gal_probs = gal_probs * bright_mask * target_cat.galaxy_bools.squeeze()
            dim_gal_probs = gal_probs * dim_mask * target_cat.galaxy_bools.squeeze()

            all_star_probs = star_probs * target_cat.star_bools.squeeze()
            bright_star_probs = star_probs * bright_mask * target_cat.star_bools.squeeze()
            dim_star_probs = star_probs * dim_mask * target_cat.star_bools.squeeze()

            for j, threshold in enumerate(thresholds):
                pred_all_gal[name][j] += (all_gal_probs > threshold).sum()
                pred_bright_gal[name][j] += (bright_gal_probs > threshold).sum()
                pred_dim_gal[name][j] += (dim_gal_probs > threshold).sum()

                pred_all_star[name][j] += (all_star_probs > threshold).sum()
                pred_bright_star[name][j] += (bright_star_probs > threshold).sum()
                pred_dim_star[name][j] += (dim_star_probs > threshold).sum()

    prop_all_gal = {}
    prop_bright_gal = {}
    prop_dim_gal = {}
    prop_all_star = {}
    prop_bright_star = {}
    prop_dim_star = {}
    for name in models:
        prop_all_gal[name] = pred_all_gal[name] / true_all_gal
        prop_bright_gal[name] = pred_bright_gal[name] / true_bright_gal
        prop_dim_gal[name] = pred_dim_gal[name] / true_dim_gal
        prop_all_star[name] = pred_all_star[name] / true_all_star
        prop_bright_star[name] = pred_bright_star[name] / true_bright_star
        prop_dim_star[name] = pred_dim_star[name] / true_dim_star

    data = {
        "prop_all_gal": prop_all_gal,
        "prop_bright_gal": prop_bright_gal,
        "prop_dim_gal": prop_dim_gal,
        "prop_all_star": prop_all_star,
        "prop_bright_star": prop_bright_star,
        "prop_dim_star": prop_dim_star,
    }
    torch.save(data, cached_path)
    return data


def plot_classification_by_threshold(prop_all, prop_bright, prop_dim, source_type, thresholds):
    """Plot classification accuracy by threshold

    Args:
        prop_all (dict): from compute_classification_probs_by_threshold
        prop_bright (dict): from compute_classification_probs_by_threshold
        prop_dim (dict): from compute_classification_probs_by_threshold
        source_type (dict): from compute_classification_probs_by_threshold
        thresholds (list): List of classification thresholds
    """
    fig, ax = plt.subplots(1, 3, figsize=(5 * 3, 5))

    ax[0].hlines([1], 0, 1, color="darkgray", linewidth=1, linestyle="dashed")
    ax[1].hlines([1], 0, 1, color="darkgray", linewidth=1, linestyle="dashed")
    ax[2].hlines([1], 0, 1, color="darkgray", linewidth=1, linestyle="dashed")

    for name in models:
        plot_config = models[name]["plot_config"]
        kwargs = {
            "color": plot_config["color"],
            "marker": plot_config["marker"],
            "markeredgecolor": "k",
            "markersize": 6,
            "linewidth": 1,
            "label": plot_config["name"],
        }
        ax[0].plot(thresholds, prop_all[name], **kwargs)
        ax[1].plot(thresholds, prop_bright[name], **kwargs)
        ax[2].plot(thresholds, prop_dim[name], **kwargs)

    ax[0].legend(fontsize="x-large")
    ax[0].set_title("All sources", fontsize="xx-large")
    ax[1].set_title(f"Bright (magnitude $<$ {BRIGHT_THRESHOLD:.2f})", fontsize="xx-large")
    ax[2].set_title(f"Faint (magnitude {FAINT_THRESHOLD:.2f}-22)", fontsize="xx-large")

    for a in ax:
        a.tick_params(axis="both", which="major", labelsize="x-large")
        a.set_xlabel("Threshold", fontsize="xx-large")
    ax[0].set_ylabel(f"Proportion of true {source_type} predicted", fontsize="xx-large")

    fig.tight_layout()
    plt.savefig(f"plots/{dataset_name}/calibration/prop_{source_type}_threshold_by_mag.pdf")


def compute_source_type_roc_curve(pred_dists, cached_path):
    """Compute the ROC curve for source type classification at different thresholds.

    Args:
        pred_dists (dict): Calibration results computed by run_calibration
        cached_path (str): Where to save the results

    Returns:
        Dict: dictionary of results
    """
    all_true, bright_true, dim_true = [], [], []
    all_pred = {name: [] for name in models}
    bright_pred = {name: [] for name in models}
    dim_pred = {name: [] for name in models}

    for i, batch in enumerate(tqdm(calib_dataloader, desc="Prob correct star/gal by threshold")):
        target_cat = TileCatalog(batch["tile_catalog"])
        target_cat = target_cat.filter_by_flux(min_flux=1.59, band=2)

        normal_mask = (target_cat.on_magnitudes(c=1)[..., 2] < 22).squeeze()
        bright_mask = (target_cat.on_magnitudes(c=1)[..., 2] < BRIGHT_THRESHOLD).squeeze()
        dim_mask = (target_cat.on_magnitudes(c=1)[..., 2] > FAINT_THRESHOLD).squeeze() * normal_mask
        on_mask = target_cat.is_on_mask.squeeze()

        true_source_type = target_cat["source_type"].squeeze()

        all_true.extend(true_source_type[on_mask * normal_mask].tolist())
        bright_true.extend(true_source_type[on_mask * bright_mask].tolist())
        dim_true.extend(true_source_type[on_mask * dim_mask].tolist())

        for name, model in models.items():
            gal_probs = pred_dists[name][i]["source_type"].probs[..., 1]

            all_pred[name].extend(gal_probs[on_mask * normal_mask])
            bright_pred[name].extend(gal_probs[on_mask * bright_mask])
            dim_pred[name].extend(gal_probs[on_mask * dim_mask])

    all_roc = {}
    bright_roc = {}
    dim_roc = {}
    for name in models:
        all_fpr, all_tpr, _ = roc_curve(all_true, all_pred[name])
        all_auc = roc_auc_score(all_true, all_pred[name])
        bright_fpr, bright_tpr, _ = roc_curve(bright_true, bright_pred[name])
        bright_auc = roc_auc_score(bright_true, bright_pred[name])
        dim_fpr, dim_tpr, _ = roc_curve(dim_true, dim_pred[name])
        dim_auc = roc_auc_score(dim_true, dim_pred[name])

        all_roc[name] = {"fpr": all_fpr, "tpr": all_tpr, "auc": all_auc}
        bright_roc[name] = {"fpr": bright_fpr, "tpr": bright_tpr, "auc": bright_auc}
        dim_roc[name] = {"fpr": dim_fpr, "tpr": dim_tpr, "auc": dim_auc}

    data = {
        "all_roc": all_roc,
        "bright_roc": bright_roc,
        "dim_roc": dim_roc,
    }
    torch.save(data, cached_path)
    return data


def plot_source_type_roc_curve(all_roc, bright_roc, dim_roc):
    """Plot ROC curve for source type classification.

    Args:
        all_roc (dict): from compute_source_type_roc_curve
        bright_roc (dict): from compute_source_type_roc_curve
        dim_roc (dict): from compute_source_type_roc_curve
    """
    fig, ax = plt.subplots(1, 3, figsize=(5 * 3, 5))

    ax[0].plot(np.linspace(0, 1, 2), color="darkgray", linewidth=1, linestyle="dashed")
    ax[1].plot(np.linspace(0, 1, 2), color="darkgray", linewidth=1, linestyle="dashed")
    ax[2].plot(np.linspace(0, 1, 2), color="darkgray", linewidth=1, linestyle="dashed")

    for name in models:
        plot_config = models[name]["plot_config"]
        kwargs = {
            "color": plot_config["color"],
            "linewidth": 1.5,
        }
        ax[0].plot(
            all_roc[name]["fpr"],
            all_roc[name]["tpr"],
            label=f"{plot_config['name']} [{all_roc[name]['auc']:.3f}]",
            **kwargs,
        )
        ax[1].plot(
            bright_roc[name]["fpr"],
            bright_roc[name]["tpr"],
            label=f"{plot_config['name']} [{bright_roc[name]['auc']:.3f}]",
            **kwargs,
        )
        ax[2].plot(
            dim_roc[name]["fpr"],
            dim_roc[name]["tpr"],
            label=f"{plot_config['name']} [{dim_roc[name]['auc']:.3f}]",
            **kwargs,
        )

    ax[0].legend(fontsize="x-large")
    ax[1].legend(fontsize="x-large")
    ax[2].legend(fontsize="x-large")

    ax[0].set_title("All sources", fontsize="xx-large")
    ax[1].set_title(f"Bright (magnitude $<$ {BRIGHT_THRESHOLD:.2f})", fontsize="xx-large")
    ax[2].set_title(f"Faint (magnitude {FAINT_THRESHOLD:.2f}-22)", fontsize="xx-large")

    for a in ax:
        a.tick_params(axis="both", which="major", labelsize="x-large")
        a.set_xlabel("Galaxy False Positive Rate", fontsize="xx-large")
    ax[0].set_ylabel(f"Galaxy True Positive Rate", fontsize="xx-large")

    fig.tight_layout()
    plt.savefig(f"plots/{dataset_name}/calibration/source_type_roc.pdf")


def compute_ci_width(pred_dists, bins, cached_path):
    """Compute average flux credible interval width and average flux standard deviation.

    Args:
        pred_dists (dict): Calibration results computed by run_calibration
        bins (list): Magnitude bins
        cached_path (str): Where to save the results

    Returns:
        Dict: dictionary of results
    """
    interval = 0.95
    tail_prob = torch.tensor((1 - interval) / 2)
    n_bins = len(bins)

    ci_width = {name: torch.zeros(n_bins) for name in models}
    ci_width_prop = {name: torch.zeros(n_bins) for name in models}
    flux_scale = {name: torch.zeros(n_bins) for name in models}
    bin_count = torch.zeros(n_bins)

    for i, batch in enumerate(tqdm(calib_dataloader, desc="CI width")):
        target_cat = TileCatalog(batch["tile_catalog"])
        target_cat = target_cat.filter_by_flux(min_flux=1.59, band=2)

        target_on_mags = target_cat.on_magnitudes(c=1)[target_cat.is_on_mask][:, 2].contiguous()
        binned_target_on_mags = torch.bucketize(target_on_mags, bins)

        bin_count += binned_target_on_mags.bincount(minlength=n_bins)

        for name, model in models.items():
            q_star_flux = pred_dists[name][i]["star_fluxes"].base_dist
            q_gal_flux = pred_dists[name][i]["galaxy_fluxes"].base_dist

            # construct equal tail intervals
            star_intervals = q_star_flux.icdf(1 - tail_prob) - q_star_flux.icdf(tail_prob)
            gal_intervals = q_gal_flux.icdf(1 - tail_prob) - q_gal_flux.icdf(tail_prob)

            # Compute CI width for true sources based on source type
            width = torch.where(
                target_cat.star_bools, star_intervals.unsqueeze(-2), gal_intervals.unsqueeze(-2)
            )
            width = width[target_cat.is_on_mask][:, 2]
            width[width == torch.inf] = 0  # temp hack to not get inf

            tmp = torch.zeros(n_bins, dtype=width.dtype)
            ci_width[name] += tmp.scatter_add(0, binned_target_on_mags, width)

            tmp = torch.zeros(n_bins, dtype=width.dtype)
            ci_width_prop[name] += tmp.scatter_add(
                0,
                binned_target_on_mags,
                width / target_cat.on_fluxes[target_cat.is_on_mask][:, 2],
            )

            # Get flux scale for true sources based on source type
            scale = torch.where(
                target_cat.star_bools,
                q_star_flux.scale.unsqueeze(-2),
                q_gal_flux.scale.unsqueeze(-2),
            )
            scale = scale[target_cat.is_on_mask][:, 2]

            scales_per_bin = torch.zeros(n_bins, dtype=scale.dtype)
            flux_scale[name] += scales_per_bin.scatter_add(0, binned_target_on_mags, scale)

    for name in models:
        ci_width[name] = ci_width[name] / bin_count
        ci_width_prop[name] = ci_width_prop[name] / bin_count
        flux_scale[name] = flux_scale[name] / bin_count
    data = {"ci_width": ci_width, "ci_width_prop": ci_width_prop, "flux_scale": flux_scale}
    torch.save(data, cached_path)
    return data


def plot_ci_width_data(data, plot_name, cfg_dict, bins):
    """Plot credible interval width and average standard deviation.

    Args:
        data (dict): from compute_ci_width
        plot_name (str): name of metric being plotted (for saving)
        cfg_dict (dict): dictionary of plot config
        bins (list): Magnitude bins
    """
    fig, ax = plt.subplots(figsize=(7.25, 5))

    for name in data:
        plot_config = models[name]["plot_config"]

        binned_vals = data[name].detach()
        kwargs = {
            "color": plot_config["color"],
            "marker": plot_config["marker"],
            "markeredgecolor": "k",
            "markersize": 6,
            "linewidth": 1,
            "label": plot_config["name"],
        }
        ax.plot(binned_vals, **kwargs)

    xticklabels = [f"{bins[i+1]:.2f}" for i in range(len(bins) - 1)]
    xticklabels.insert(0, f"$<${bins[0]:.2f}")
    ax.set_xticks(range(len(xticklabels)), xticklabels)

    ax.tick_params(axis="both", which="major", labelsize="xx-large")
    ax.set_xlabel("r-band Magnitude", fontsize="xx-large")
    ax.set_ylabel(cfg_dict["ylabel"], fontsize=cfg_dict["ylabel_size"])
    ax.legend(fontsize="x-large")

    fig.tight_layout()
    plt.savefig(f"plots/{dataset_name}/calibration/{plot_name}_by_mag.pdf")


#################################################
# Evaluate models
#################################################
if args.run_eval:
    print("Computing metrics for eval")
    run_eval()


#################################################
# Plot eval results
#################################################
if args.plot_eval:
    print("Plotting eval results")
    # Load saved results
    cached_path = f"data/{dataset_name}/metrics.pt"
    assert os.path.exists(
        cached_path
    ), f"ERROR: could not find cached metrics at {cached_path}. Try running with the --run_eval flag."
    data = torch.load(f"data/{dataset_name}/metrics.pt")

    # Choose metrics to plot and specify labels, marker, and color for each model
    metrics_to_plot = {
        "detection_precision": {
            "ylabel": "Precision",
            "metric_class": "detection_performance",
            "yaxis_in_percent": False,
        },
        "detection_recall": {
            "ylabel": "Recall",
            "metric_class": "detection_performance",
            "yaxis_in_percent": False,
        },
        "detection_f1": {
            "ylabel": "F1-Score",
            "metric_class": "detection_performance",
            "yaxis_in_percent": False,
        },
        "classification_acc": {
            "ylabel": "Classification Accuracy",
            "metric_class": "source_type_accuracy",
            "yaxis_in_percent": False,
        },
        "flux_err_r_mpe": {
            "ylabel": "r-band Flux Mean \% Error",
            "metric_class": "flux_error",
            "yaxis_in_percent": True,
        },
        "flux_err_r_mape": {
            "ylabel": "r-band Flux Mean Abosolute \% Error",
            "metric_class": "flux_error",
            "yaxis_in_percent": True,
        },
        "galaxy_disk_frac_mae": {
            "ylabel": "Disk fraction of flux MAE",
            "metric_class": "gal_shape_error",
            "yaxis_in_percent": False,
        },
        "galaxy_beta_radians_mae": {
            "ylabel": "Angle MAE",
            "metric_class": "gal_shape_error",
            "yaxis_in_percent": False,
        },
        "galaxy_disk_q_mae": {
            "ylabel": "Disk minor:major ratio MAE",
            "metric_class": "gal_shape_error",
            "yaxis_in_percent": False,
        },
        "galaxy_a_d_mae": {
            "ylabel": "Disk major axis MAE",
            "metric_class": "gal_shape_error",
            "yaxis_in_percent": False,
        },
        "galaxy_bulge_q_mae": {
            "ylabel": "Bulge minor:major ratio MAE",
            "metric_class": "gal_shape_error",
            "yaxis_in_percent": False,
        },
        "galaxy_a_b_mae": {
            "ylabel": "Bulge major axis MAE",
            "metric_class": "gal_shape_error",
            "yaxis_in_percent": False,
        },
        "galaxy_disk_hlr_mae": {
            "ylabel": "Disk HLR MAE",
            "metric_class": "gal_shape_error",
            "yaxis_in_percent": False,
        },
        "galaxy_bulge_hlr_mae": {
            "ylabel": "Bulge HLR MAE",
            "metric_class": "gal_shape_error",
            "yaxis_in_percent": False,
        },
    }

    # Plot!
    os.makedirs(f"plots/{dataset_name}/metrics", exist_ok=True)
    for metric, metric_cfg in metrics_to_plot.items():
        plot_metric(data, metric, metric_cfg)


#################################################
# Run calibration
#################################################
if args.run_calibration:
    print("Computing calibration results")
    run_calibration()


#################################################
# Plot calibration results
#################################################
if args.plot_calibration:
    print("Plotting calibration results")
    cached_path = (
        f"/home/aakashdp/bliss/case_studies/psf_variation/data/{dataset_name}/posterior_dists.pt"
    )
    assert os.path.exists(
        cached_path
    ), f"ERROR: could not find cached calibration data at {cached_path}. Try running with the --run_calibration flag."
    pred_dists = torch.load(cached_path)
    os.makedirs(f"plots/{dataset_name}/calibration", exist_ok=True)

    ### Expected number of sources
    bins = torch.arange(20)
    cached_path = f"data/{dataset_name}/true_vs_pred_sources.pt"
    if os.path.exists(cached_path):
        print(f"Loading cached data from {cached_path}")
        source_data = torch.load(cached_path)
    else:
        source_data = compute_expected_sources(pred_dists, bins, cached_path)

    mean_sources_per_bin = source_data["mean_sources_per_bin"]
    mean_bright_per_bin = source_data["mean_bright_per_bin"]
    mean_dim_per_bin = source_data["mean_dim_per_bin"]

    plot_expected_vs_predicted_sources(mean_sources_per_bin, mean_bright_per_bin, mean_dim_per_bin)

    ### Probability predicted magnitude is within x of true magnitude
    bins = torch.tensor(
        [17.777, 19.101, 19.781, 20.258, 20.625, 20.940, 21.227, 21.495, 21.746, 22.000]
    )
    cached_path = f"data/{dataset_name}/prob_flux_within_one_mag.pt"
    if os.path.exists(cached_path):  # Load cached data if exists
        print(f"Loading cached data from {cached_path}")
        binned_avg_flux_probs = torch.load(cached_path)
    else:
        binned_avg_flux_probs = compute_prob_flux_within_one_mag(pred_dists, bins, cached_path)

    plot_flux_within_one_mag(binned_avg_flux_probs, bins)

    ### Proportion of true fluxes in credible interval
    intervals = torch.linspace(0.5, 1, 11)
    cached_path = f"data/{dataset_name}/prop_flux_in_interval.pt"
    if os.path.exists(cached_path):
        print(f"Loading cached data from {cached_path}")
        data = torch.load(cached_path)
    else:
        data = compute_prop_flux_in_interval(pred_dists, intervals, cached_path)

    prop_all_in_eti = data["prop_all_in_eti"]
    prop_bright_in_eti = data["prop_bright_in_eti"]
    prop_dim_in_eti = data["prop_dim_in_eti"]

    plot_prop_flux_in_interval(prop_all_in_eti, prop_bright_in_eti, prop_dim_in_eti, intervals)

    ### Prob source type by magnitude
    bins = torch.tensor(
        [17.777, 19.101, 19.781, 20.258, 20.625, 20.940, 21.227, 21.495, 21.746, 22.000]
    )
    cached_path = f"data/{dataset_name}/true_source_type_probs.pt"
    if os.path.exists(cached_path):
        print(f"Loading cached data from {cached_path}")
        binned_source_type_probs = torch.load(cached_path)
    else:
        binned_source_type_probs = compute_avg_prob_true_source_type(pred_dists, bins, cached_path)

    plot_prob_true_source_type(binned_source_type_probs, bins)

    ### Prob correct galaxy / star by threshold
    thresholds = torch.linspace(0, 1, 11)
    thresholds[-1] = 0.99  # 1 is trivial - 0.99 gives more information
    cached_path = f"data/{dataset_name}/source_type_classification_by_threshold.pt"
    if os.path.exists(cached_path):
        print(f"Loading cached data from {cached_path}")
        data = torch.load(cached_path)
    else:
        data = compute_classification_probs_by_threshold(pred_dists, thresholds, cached_path)

    prop_all_gal = data["prop_all_gal"]
    prop_bright_gal = data["prop_bright_gal"]
    prop_dim_gal = data["prop_dim_gal"]
    prop_all_star = data["prop_all_star"]
    prop_bright_star = data["prop_bright_star"]
    prop_dim_star = data["prop_dim_star"]

    # Plot gal
    plot_classification_by_threshold(
        prop_all_gal, prop_bright_gal, prop_dim_gal, "galaxies", thresholds
    )

    # Plot star
    plot_classification_by_threshold(
        prop_all_star, prop_bright_star, prop_dim_star, "stars", thresholds
    )

    ## Source type classification ROC curve
    cached_path = (
        f"/home/aakashdp/bliss/case_studies/psf_variation/data/{dataset_name}/source_type_roc.pt"
    )
    if os.path.exists(cached_path):
        print(f"Loading cached data from {cached_path}")
        data = torch.load(cached_path)
    else:
        data = compute_source_type_roc_curve(pred_dists, cached_path)

    all_roc = data["all_roc"]
    bright_roc = data["bright_roc"]
    dim_roc = data["dim_roc"]

    plot_source_type_roc_curve(all_roc, bright_roc, dim_roc)

    ### CI width / standard deviation vs magnitude
    bins = torch.tensor(
        [17.777, 19.101, 19.781, 20.258, 20.625, 20.940, 21.227, 21.495, 21.746, 22.000]
    )
    cached_path = f"data/{dataset_name}/ci_width_and_flux_scale.pt"
    if os.path.exists(cached_path):
        print(f"Loading cached data from {cached_path}")
        data = torch.load(cached_path)
    else:
        data = compute_ci_width(pred_dists, bins, cached_path)

    ci_width = data["ci_width"]
    ci_width_prop = data["ci_width_prop"]
    flux_scale = data["flux_scale"]

    metrics_to_plot = {
        "ci_width": {"ylabel": "Average width of 95\% CI (nmgy)", "ylabel_size": "xx-large"},
        "ci_width_prop": {
            "ylabel": "Average width of 95\% CI / true flux",
            "ylabel_size": "xx-large",
        },
        "flux_scale": {"ylabel": "Average $\\sigma$ for predicted flux", "ylabel_size": "xx-large"},
    }

    for metric, cfg_dict in metrics_to_plot.items():
        plot_ci_width_data(eval(metric), metric, cfg_dict, bins)
