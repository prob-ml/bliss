import argparse
import copy
from os import environ
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import torch
from hydra import compose, initialize
from hydra.utils import instantiate
from matplotlib import pyplot as plt

from bliss.catalog import TileCatalog
from bliss.metrics import BlissMetrics, MetricsMode

# Usage: python3 ablation_experiment.py fp1 [path1] fp2 [path2] b1 [bands1] b2 [bands2]
# \ [-mean_sources mean_sources] [-llf log_low_flux] [-lhf log_high_flux] [-b bins]


def nmgy_to_nelec_for_catalog(est_cat, nelec_per_nmgy_per_band):
    fluxes_suffix = "_fluxes"
    # reshape nelec_per_nmgy_per_band to (1, 1, 1, 1, {n_bands}) to broadcast
    nelec_per_nmgy_per_band = torch.tensor(nelec_per_nmgy_per_band, device=est_cat.device)
    nelec_per_nmgy_per_band = nelec_per_nmgy_per_band.view(1, 1, 1, 1, -1)
    for key in est_cat.keys():
        if key.endswith(fluxes_suffix):
            est_cat[key] = est_cat[key] * nelec_per_nmgy_per_band
    return est_cat


def pred(cfg, tile_cat, images, background):
    """Compute predictions using BLISS predict pipeline.

    Args:
        cfg: config file
        tile_cat: true tile catalog (not cropped)
        images: simulated images
        background: simulated background

    Returns:
        est_tile_cat: estimated tile catalog
    """
    conf = cfg.copy()
    imgs = copy.deepcopy(images)
    bgs = copy.deepcopy(background)
    survey = instantiate(cfg.predict.dataset, load_image_data=True)
    survey_objs = [survey[i] for i in range(len(survey))]
    encoder = instantiate(conf.encoder)
    enc_state_dict = torch.load(conf.predict.weight_save_path)
    encoder.load_state_dict(enc_state_dict)
    encoder.eval()
    batch = {"images": imgs, "background": bgs}
    out_dict = encoder.predict_step(batch, None)
    est_cat = out_dict["est_cat"]
    nelec_per_nmgy_per_band = np.mean(survey_objs[0]["flux_calibration_list"], axis=1)
    return nmgy_to_nelec_for_catalog(est_cat, nelec_per_nmgy_per_band)


def create_df(key, d, steps):
    """Construct detection metrics dataframe for plotting.

    Args:
        d: dictionary with following keys:
            tgs:
            tss:
            egs:
            ess:
            boot_precision_25
            boot_precision_75
            boot_recall_25
            boot_recall_75
            prec
            rec
        key: model within dictionary to create DF for
        steps: steps for bootstrapping

    Returns:
        df: Pandas dataframe containing above data
    """
    metrics = {}
    if "tgs" not in metrics:
        metrics["tgs"] = []
    for i in len(d[key]["tgs"]):
        metrics["tgs"].append(i.numpy())
    if "tss" not in metrics:
        metrics["tss"] = []
    for i in len(d[key]["tss"]):
        metrics["tss"].append(i.numpy())
    if "egs" not in metrics:
        metrics["egs"] = []
    for i in len(d[key]["egs"]):
        metrics["egs"].append(i.numpy())
    if "ess" not in metrics:
        metrics["ess"] = []
    for i in len(d[key]["ess"]):
        metrics["ess"].append(i.numpy())

    metrics["boot_precision_25"] = d[key]["boot_precision_25"]
    metrics["boot_precision_75"] = d[key]["boot_precision_75"]
    metrics["boot_recall_25"] = d[key]["boot_recall_25"]
    metrics["boot_recall_75"] = d[key]["boot_recall_75"]
    metrics["prec"] = d[key]["prec"]
    metrics["rec"] = d[key]["rec"]

    df = pd.DataFrame(metrics)
    df["Max R-Flux (Log-10)"] = steps
    return df.rename(
        columns={"detection_precision": "Precision", "detection_recall": "Recall", "f1": "F-1"}
    )


def construct_pred_dict(cfg, fpaths, bands, ttc, imgs, bgs):
    d = {}
    for i, fpath in enumerate(fpaths):
        d[fpath] = {"bands": bands[i], "est_tile_cat": None}
        conf = cfg.copy()
        conf.encoder.bands = bands[i]
        conf.predict.weight_save_path = fpath
        d[fpath]["est_tile_cat"] = pred(conf, ttc, imgs, bgs)
    return d


def detection_dfs(d, steps, true_tile_cat):  # pylint: disable=R0915
    """Creates dataframes for each tested model populated with detection metrics.

    Args:
        d: dictionary output from <construct_pred_dict>
        steps: vector containing bin start points
        true_tile_cat: true tile catalog used for computing model predictions

    Returns:
        df_d: dictionary containing each model's detection dataframe
    """

    bins = np.array([np.array([steps[i], steps[i + 1]]) for i in range(len(steps) - 1)])

    for model in d.keys():
        tc_cat_est = d[model]["est_tile_cat"]
        tc_cat_true = true_tile_cat
        d[model]["tgs"] = []
        d[model]["tss"] = []
        d[model]["egs"] = []
        d[model]["ess"] = []
        d[model]["boot_precision"] = []
        d[model]["boot_recall"] = []
        d[model]["prec"] = []
        d[model]["rec"] = []
        # for j in range(len(steps[:-1])):
        for b1, b2 in bins:
            # COMPUTE RECALL
            tc_cat_true_mod = tc_cat_true.filter_tile_catalog_by_flux(b1, b2)
            tc_cat_est_mod = TileCatalog(4, tc_cat_est.to_dict())
            tc_cat_est_mod.n_sources = tc_cat_true_mod.n_sources
            d[model]["rec"].append(metrics(tc_cat_true_mod, tc_cat_est_mod)["detection_recall"])
            # COMPUTE PRECISION
            tc_cat_est_mod = tc_cat_est.filter_tile_catalog_by_flux(b1, b2)
            tc_cat_true_mod = TileCatalog(4, tc_cat_true.to_dict())
            tc_cat_true_mod.n_sources = tc_cat_est_mod.n_sources
            d[model]["prec"].append(metrics(tc_cat_true_mod, tc_cat_est_mod)["detection_precision"])
            d[model]["tgs"].append(tc_cat_true_mod.galaxy_bools.sum())
            d[model]["tss"].append(tc_cat_true_mod.star_bools.sum())
            d[model]["egs"].append(tc_cat_est_mod.galaxy_bools.sum())
            d[model]["ess"].append(tc_cat_est_mod.star_bools.sum())

        # Perform bootstrapping
        n_boots = 10
        n_matches = len(tc_cat_true["star_fluxes"])
        n_bins = len(steps) - 1
        boot_precision = np.zeros((n_boots, n_bins))
        boot_recall = np.zeros((n_boots, n_bins))

        boot_indices = np.random.randint(0, n_matches, (n_boots, n_matches))

        # compute boostrap precision and recall per bin
        for ii in range(n_boots):
            tc_cat_true_boot = copy.deepcopy(tc_cat_true)
            tc_cat_est_boot = TileCatalog(4, tc_cat_est.to_dict())
            d_true = {}
            d_est = {}
            for key, _ in tc_cat_true_boot.to_dict().items():
                d_true[key] = tc_cat_true[key][boot_indices[ii]]
                d_est[key] = tc_cat_est[key][boot_indices[ii]]
            tc_cat_true_boot = TileCatalog(4, d_true)
            tc_cat_est_boot = TileCatalog(4, d_est)
            for jj, (b1, b2) in enumerate(bins):
                # COMPUTE BOOT RECALL
                tc_cat_true_boot_bin = copy.deepcopy(tc_cat_true_boot)
                tc_cat_true_boot_bin = tc_cat_true_boot.filter_tile_catalog_by_flux(b1, b2)
                tc_cat_est_boot_bin = TileCatalog(4, tc_cat_est_boot.to_dict())
                tc_cat_est_boot_bin.n_sources = tc_cat_true_boot_bin.n_sources
                m = metrics(tc_cat_true_boot_bin, tc_cat_est_boot_bin)
                r = m["detection_recall"]
                # COMPUTE BOOT PRECISION
                tc_cat_est_boot_bin = tc_cat_est_boot.filter_tile_catalog_by_flux(b1, b2)
                tc_cat_true_boot_bin = TileCatalog(4, tc_cat_true_boot.to_dict())
                tc_cat_true_boot_bin.n_sources = tc_cat_est_boot_bin.n_sources
                m = metrics(tc_cat_true_boot_bin, tc_cat_est_boot_bin)
                p = m["detection_precision"]

                boot_precision[ii][jj] = p
                boot_recall[ii][jj] = r
        d[model]["boot_precision"].append(boot_precision)
        d[model]["boot_recall"].append(boot_recall)

    for model in d.keys():
        boot_precision = d[model]["boot_precision"].pop()
        boot_recall = d[model]["boot_recall"].pop()
        d[model]["boot_precision_25"] = []
        d[model]["boot_recall_25"] = []
        d[model]["boot_precision_75"] = []
        d[model]["boot_recall_75"] = []

        for i in range(len(bins)):  # noqa: WPS518
            d[model]["boot_precision_25"].append(np.quantile(boot_precision[:, i], 0.25))
            d[model]["boot_recall_25"].append(np.quantile(boot_recall[:, i], 0.25))
            d[model]["boot_precision_75"].append(np.quantile(boot_precision[:, i], 0.75))
            d[model]["boot_recall_75"].append(np.quantile(boot_recall[:, i], 0.75))

    out_d = {}

    for model in d.keys():
        out_d[model] = create_df(model, d, np.log10(steps[1:]))

    return out_d


def plot_ablation(dfs, names):  # pylint: disable=R0915
    """Creates ablation figure plot from multiband paper.

    Args:
        dfs: list, contains dataframes outputted from detection_dfs.
        names: list, contains strings to be used as labels for plot.

    Returns:
        fig: Figure, Ablation figure
    """

    font = {
        "family": "serif",
        "weight": "heavy",
        "size": 15,
    }

    matplotlib.rc("font", **font)
    fig, ((ax1, ax3), (ax2, ax4)) = plt.subplots(
        2, 2, figsize=(16, 8), gridspec_kw={"height_ratios": [1, 2]}, sharex=True
    )
    for i, df in enumerate(dfs):
        # params
        xlabel = "Log-10 R-Flux"
        metric_type = "Precision"
        legend_size_hist = 10
        where_step = "mid"
        n_ticks = 5
        ordmag = 3

        prec = df["prec"]
        precision1 = df["boot_precision_25"]
        precision2 = df["boot_precision_75"]
        recall = df["rec"]

        tgcount = df["tgs"]
        tscount = df["tss"]
        egcount = df["egs"]
        escount = df["ess"]

        ymin = min(prec.min(), recall.min())
        yticks = np.arange(np.round(ymin, 1), 1.1, 0.1)
        ax2.set_yticks(yticks)
        ax2.plot(np.log10(steps[1:]), prec, "-o", label=f"{names[i]}", markersize=6)
        ax2.fill_between(np.log10(steps[1:]), precision1, precision2, alpha=0.5)
        ax2.legend(loc="best")
        ax2.set_xlabel(xlabel)
        ax2.set_ylabel(f"{metric_type}")
        ax2.grid(linestyle="-", linewidth=0.5, which="major", axis="both")

    # setup histogram plot up top
    c1 = plt.rcParams["axes.prop_cycle"].by_key()["color"][3]
    c2 = plt.rcParams["axes.prop_cycle"].by_key()["color"][4]
    ax1.step(np.log10(steps[1:]), tgcount, label="True galaxies", where=where_step, color=c1)
    ax1.step(np.log10(steps[1:]), tscount, label="True stars", where=where_step, color=c2)
    ax1.step(
        np.log10(steps[1:]), egcount, label="Pred. galaxies", ls="--", where=where_step, color=c1
    )
    ax1.step(np.log10(steps[1:]), escount, label="Pred. stars", ls="--", where=where_step, color=c2)
    ymax = max(tgcount.max(), tscount.max(), egcount.max(), escount.max())
    yticks = np.round(np.linspace(0, ymax, n_ticks), -ordmag)
    ax1.set_yticks(yticks)
    ax1.set_ylabel("Counts")
    ax1.legend(loc="best", prop={"size": legend_size_hist})
    ax1.grid(linestyle="-", linewidth=0.5, which="major", axis="both")
    plt.subplots_adjust(hspace=0)

    for i, df in enumerate(dfs):
        # params
        xlabel = "Log-10 R-Flux"
        metric_type = "Recall"
        legend_size_hist = 10
        where_step = "mid"
        n_ticks = 5
        ordmag = 3

        recall = df["rec"]
        recall1 = df["boot_recall_25"]
        recall2 = df["boot_recall_75"]
        prec = df["prec"]

        tgcount = df["tgs"]
        tscount = df["tss"]
        egcount = df["egs"]
        escount = df["ess"]

        # (bottom) plot of precision and recall
        ymin = min(prec.min(), recall.min())
        yticks = np.arange(np.round(ymin, 1), 1.1, 0.1)
        ax4.set_yticks(yticks)
        ax4.plot(np.log10(steps[1:]), recall, "-o", label=f"{names[i]}", markersize=6)
        ax4.fill_between(np.log10(steps[1:]), recall1, recall2, alpha=0.5)
        ax4.legend(loc="best")
        ax4.set_xlabel(xlabel)
        ax4.set_ylabel(f"{metric_type}")
        ax4.grid(linestyle="-", linewidth=0.5, which="major", axis="both")

    # setup histogram plot up top
    c1 = plt.rcParams["axes.prop_cycle"].by_key()["color"][3]
    c2 = plt.rcParams["axes.prop_cycle"].by_key()["color"][4]
    ax3.step(np.log10(steps[1:]), tgcount, label="True galaxies", where=where_step, color=c1)
    ax3.step(np.log10(steps[1:]), tscount, label="True stars", where=where_step, color=c2)
    ax3.step(
        np.log10(steps[1:]), egcount, label="Pred. galaxies", ls="--", where=where_step, color=c1
    )
    ax3.step(np.log10(steps[1:]), escount, label="Pred. stars", ls="--", where=where_step, color=c2)
    ymax = max(tgcount.max(), tscount.max(), egcount.max(), escount.max())
    yticks = np.round(np.linspace(0, ymax, n_ticks), -ordmag)
    ax3.set_yticks(yticks)
    ax3.set_ylabel("Counts")
    ax3.legend(loc="best", prop={"size": legend_size_hist})
    ax3.grid(linestyle="-", linewidth=0.5, which="major", axis="both")
    plt.subplots_adjust(hspace=0)
    plt.savefig("ablation.png")

    return fig


# --

CLI = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
CLI.add_argument(
    "fpath1",
    nargs=1,
    type=str,
    default="",
    help="String containing full path to first model state dict.",
)
CLI.add_argument(
    "fpath2",
    nargs=1,
    type=str,
    default="",
    help="String containing full path to second model state dict.",
)
CLI.add_argument(
    "-bands1",
    nargs=1,
    type=int,
    default=[1],
    help="Number of taken as input by each model specified in fnames.",
)
CLI.add_argument(
    "-bands2",
    nargs=1,
    type=int,
    default=[3],
    help="Number of taken as input by each model specified in fnames.",
)
CLI.add_argument(
    "-ms",
    "--mean_sources",
    type=float,
    default=0.02,
    help="Mean sources argument for simulated data.",
)
CLI.add_argument(
    "-llf",
    "--log_low_flux",
    nargs=1,
    type=int,
    default=3,
    help="Low-end range for flux bins (log-scale).",
)
CLI.add_argument(
    "-lhf",
    "--log_high_flux",
    nargs=1,
    type=int,
    default=6,
    help="High-end range for flux bins (log-scale).",
)
CLI.add_argument(
    "-b",
    "--bins",
    nargs=1,
    type=int,
    default=12,
    help="Number of bins for binned flux experiment.",
)

args = CLI.parse_args()
conf = vars(args)  # noqa: WPS421
print("conf: ", conf)  # noqa: WPS421
fp1 = conf["fpath1"]
fp2 = conf["fpath2"]
bands1 = conf["bands1"]
bands2 = conf["bands2"]
mean_sources = conf["mean_sources"]
log_low_flux = conf["log_low_flux"]
log_high_flux = conf["log_high_flux"]
bins = conf["bins"]

fps = fp1 + fp2
n_bands = bands1 + bands2

if bands1 == [1]:
    bands1 = [2]
elif bands1 == [3]:
    bands1 = [1, 2, 3]
else:
    bands1 = [0, 1, 2, 3, 4]
if bands2 == [1]:
    bands2 = [2]
elif bands2 == [3]:
    bands2 = [1, 2, 3]
else:
    bands2 = [0, 1, 2, 3, 4]

bands = [bands1, bands2]

environ["BLISS_HOME"] = str(Path().resolve().parents[1])
with initialize(config_path=".", version_base=None):
    cfg = compose("config")

# adjust mean sources based on CLI
cfg.prior.mean_sources = mean_sources

# Generate
print("Simulating data...")  # noqa: WPS421
cfg.prior.batch_size = 128
sim = instantiate(cfg.simulator)
tc = sim.catalog_prior.sample()
image_ids, image_id_indices = sim.randomized_image_ids(sim.catalog_prior.batch_size)
images, background, deconv, psf_params, tile_c = sim.simulate_image(tc, image_ids, image_id_indices)

# Crop true tile catalog
true_tile_cat = tile_c.symmetric_crop(cfg.encoder.tiles_to_crop)

print("Finished simulating. Making predictions...")  # noqa: WPS421
# construct prediction dictionaries for each model given freshly simulated data
d = construct_pred_dict(cfg, fps, bands, true_tile_cat, images, background)

# Generate bin-intervals for histogram
steps = np.logspace(3, 6, num=12)  # fluxes

# Instantiate metrics object
metrics = BlissMetrics(survey_bands=[0, 1, 2, 3, 4], mode=MetricsMode.TILE)

print("Predictions made. Constructing DFs...")  # noqa: WPS421
out_d = detection_dfs(d, steps, true_tile_cat)
model1_df, model2_df = list(out_d.values())  # pylint:disable=W0632

# pack dfs into array
dfs = [model1_df, model2_df]
names = ["Single-Band", "Two-Band", "Three-Band", "Four-Band", "Five-Band"]
plot_bands = [names[i] for i in n_bands]

print("DFs constructed. Plotting...")  # noqa: WPS421
# fig saved to 'ablation.png' in multiband directory
fig = plot_ablation(dfs, plot_bands)
print("Complete!")  # noqa: WPS421
