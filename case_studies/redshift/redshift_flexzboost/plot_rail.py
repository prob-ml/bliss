# %% imports
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from hydra import compose, initialize
from matplotlib import pyplot as plt

# %% configs
with initialize(config_path="../", version_base=None):
    cfg = compose("artifact_creation")
rail_dir = Path(cfg.paths["rail_checkpoints"])

# %% load csvs
results = [
    pd.read_parquet(rail_dir / f"flexzboost_predictions_split_{x}.parquet") for x in range(7)
]
result_normal_boost = pd.read_parquet(rail_dir / "normalboost_predictions_split_0.parquet")

########################################################################
# %% scatter z_true vs z_pred_L2
plt.gcf().set_dpi(300)
fig = plt.figure(constrained_layout=True, figsize=(15, 5))
# Create a grid with 1 row and 10 columns (3+3+3+1 layout)
gs = fig.add_gridspec(nrows=1, ncols=10)

ax0 = fig.add_subplot(gs[0, 0:3])
ax1 = fig.add_subplot(gs[0, 3:6])
ax2 = fig.add_subplot(gs[0, 6:9])
cax = fig.add_subplot(gs[0, 9])

axes = [ax0, ax1, ax2]

plots_info = [
    {
        "pred": "z_pred_L2",
        "ylabel": "FlexZBoost Posterior Mean",
        "title": "True vs Posterior Mean",
    },
    {
        "pred": "z_pred_outlier",
        "ylabel": "FlexZBoost Outlier-Minimizing Prediction",
        "title": "True vs Posterior Outlier-Minimizer",
    },
    {
        "pred": "z_pred_mode",
        "ylabel": "FlexZBoost Mode",
        "title": "True vs Posterior Mode",
    },
]

# Store the mappable from the first histogram to use for a shared colorbar.

y = results[0]["z_true"]
mappable = None
for i, info in enumerate(plots_info):
    ax = axes[i]
    yhat = results[0][info["pred"]]
    h = ax.hist2d(
        y,
        yhat,
        bins=250,
        range=[[0, 3], [0, 3]],
        cmap="plasma",
        norm=matplotlib.colors.LogNorm(vmin=1, vmax=5000),
    )
    if mappable is None:
        mappable = h[3]

    # Plot the diagonal line
    ax.plot([0, 3], [0, 3], "k--", lw=2)

    x = np.linspace(0, 3, 1000)
    y_upper = x + 0.15 * (1 + x)
    y_lower = x - 0.15 * (1 + x)
    ax.plot(x, y_upper, "r--", lw=1, label="Outlier threshold")
    ax.plot(x, y_lower, "r--", lw=1)
    outlier_count = np.mean(np.abs(y - yhat) / (1 + y) > 0.15)
    mse = ((y - yhat) ** 2).mean()
    ax.legend()
    ax.set_title(f"{info['title']}\n MSE: {mse:.3f} \n Outlier Proportion: {outlier_count:.3f}")

    ax.set_xlabel("True Redshift")
    ax.set_ylabel(info["ylabel"])
    ax.set_xlim(-0.01, 3.01)
    ax.set_ylim(-0.01, 3.01)
    ax.set_aspect("equal")

# Add a shared colorbar in the last grid cell
cbar = fig.colorbar(mappable, cax=cax, label="# sources")
cbar.ax.set_aspect(5)  # Increase the aspect ratio to make the colorbar narrower

########################################################################
# %% plot PIT line

plt.gcf().set_dpi(300)
plt.figure(figsize=(4, 2))

pit_values = results[0]["pit_values"]
sorted_pit_values = sorted(pit_values)
uniform_cdf = [i / len(pit_values) for i in range(len(pit_values))]

# Plot the difference between PIT and Uniform
difference = [u - p for p, u in zip(sorted_pit_values, uniform_cdf)]
plt.plot(sorted_pit_values, difference, label="Difference (PIT - Uniform)")

plt.plot([0, 1], [0, 0], "k--", lw=2, label="Uniform CDF")

plt.xlabel("PIT Value")
plt.ylabel("PIT Calibration")
plt.title("PIT for flexzboost")
plt.legend()
plt.grid(True)

########################################################################
# %% plot PIT line for boost

plt.gcf().set_dpi(300)
plt.figure(figsize=(4, 2))

pit_values = result_normal_boost["pit_values"]
sorted_pit_values = sorted(pit_values)
uniform_cdf = [i / len(pit_values) for i in range(len(pit_values))]

# Plot the difference between PIT and Uniform
difference = [u - p for p, u in zip(sorted_pit_values, uniform_cdf)]
plt.plot(sorted_pit_values, difference, label="Difference (PIT - Uniform)")

plt.plot([0, 1], [0, 0], "k--", lw=2, label="Uniform CDF")

plt.xlabel("PIT Value")
plt.ylabel("PIT Calibration")
plt.title("PIT for gaussian trained with boost")
plt.legend()
plt.grid(True)

########################################################################
# %% bin by redshift plots
plt.gcf().set_dpi(300)
# now 5 rows: counts, L1, bias, CRPS, and max PIT deviation
fig, axs = plt.subplots(
    5, 1, figsize=(8, 9), sharex=True, gridspec_kw={"height_ratios": [1, 1, 1, 1, 1.2]}
)

redshift_bins = np.r_[0:2.5:10j]

# FlexZBoost splits in C0
for split_idx in range(7):
    y = results[split_idx]["z_true"]
    yhat = results[split_idx]["z_pred_L2"]
    samp1 = results[split_idx]["z_pred_sample1"]
    samp2 = results[split_idx]["z_pred_sample2"]
    pit = results[split_idx]["pit_values"]

    counts, l1_err, bias, crps, max_pit_dev = [], [], [], [], []
    for j in range(len(redshift_bins) - 1):
        lo, hi = redshift_bins[j], redshift_bins[j + 1]
        mask = (yhat >= lo) & (yhat < hi)
        y_b, yh_b = y[mask], yhat[mask]
        s1, s2, p_b = samp1[mask], samp2[mask], pit[mask]

        counts.append(len(y_b))
        l1_err.append(np.abs(y_b - yh_b).mean() if len(y_b) else np.nan)
        bias.append(((yh_b - y_b) / (1 + y_b)).mean() if len(y_b) else np.nan)

        if len(y_b):
            m1 = np.mean(np.abs(y_b - s1))
            m2 = np.mean(np.abs(y_b - s2))
            m3 = np.mean(np.abs(s1 - s2))
            crps.append(0.5 * (m1 + m2) - 0.5 * m3)
        else:
            crps.append(np.nan)

        if len(p_b):
            D = 1.36 / np.sqrt(len(p_b))
            p_sorted = np.sort(p_b)
            uniform = np.arange(len(p_b)) / len(p_b)
            dev = np.abs(uniform - p_sorted).max()
            max_pit_dev.append(dev if dev > D else np.nan)
        else:
            max_pit_dev.append(np.nan)

    centers = (redshift_bins[:-1] + redshift_bins[1:]) / 2
    axs[0].bar(
        centers,
        counts,
        width=np.diff(redshift_bins),
        alpha=0.3,
        color="C0",
        label=f"flexzboost split {split_idx}",
    )
    axs[1].plot(centers, l1_err, marker="o", color="C0")
    axs[2].plot(centers, bias, marker="o", color="C0")
    axs[3].plot(centers, crps, marker="o", color="C0")
    axs[4].plot(centers, max_pit_dev, marker="o", color="C0")

# Now include the single Gaussian-boost result in C1
y = result_normal_boost["z_true"]
yhat = result_normal_boost["z_pred_L2"]
samp1 = result_normal_boost["z_pred_sample1"]
samp2 = result_normal_boost["z_pred_sample2"]
pit = result_normal_boost["pit_values"]

counts_b, l1_b, bias_b, crps_b, max_pit_b = [], [], [], [], []
for j in range(len(redshift_bins) - 1):
    lo, hi = redshift_bins[j], redshift_bins[j + 1]
    mask = (yhat >= lo) & (yhat < hi)
    y_b, yh_b = y[mask], yhat[mask]
    s1 = samp1[mask]
    s2 = samp2[mask]
    p_b = pit[mask]

    counts_b.append(len(y_b))
    l1_b.append(np.abs(y_b - yh_b).mean() if len(y_b) else np.nan)
    bias_b.append(((yh_b - y_b) / (1 + y_b)).mean() if len(y_b) else np.nan)

    if len(y_b):
        m1 = np.mean(np.abs(y_b - s1))
        m2 = np.mean(np.abs(y_b - s2))
        m3 = np.mean(np.abs(s1 - s2))
        crps_b.append(0.5 * (m1 + m2) - 0.5 * m3)
    else:
        crps_b.append(np.nan)

    if len(p_b):
        D = 1.36 / np.sqrt(len(p_b))
        p_sorted = np.sort(p_b)
        uniform = np.arange(len(p_b)) / len(p_b)
        dev = np.abs(uniform - p_sorted).max()
        max_pit_b.append(dev if dev > D else np.nan)
    else:
        max_pit_b.append(np.nan)

# plot boost results
# axs[0].bar(centers, counts_b, width=np.diff(redshift_bins), alpha=0.3,
#            color="C1", label="normal boost")
axs[1].plot(centers, l1_b, marker="s", color="C1")
axs[2].plot(centers, bias_b, marker="s", color="C1")
axs[3].plot(centers, crps_b, marker="s", color="C1")
axs[4].plot(centers, max_pit_b, marker="s", color="C1")

# labels and styling
axs[0].set_ylabel("# Sources")
axs[0].set_title(r"Metrics by redshift bin (binned on $\hat y$)")
axs[1].set_ylabel("L1 Error")
axs[2].set_ylabel("Bias")
axs[3].set_ylabel("CRPS")
axs[4].set_ylabel("KS(PIT,Uniform)")
axs[4].set_xlabel("Posterior Mean Redshift Bin")

# x‐axis ticks
tick_labels = [
    f"{redshift_bins[i]:.1f}-{redshift_bins[i+1]:.1f}" for i in range(len(redshift_bins) - 1)
]
axs[4].set_xticks(centers)
axs[4].set_xticklabels(tick_labels, rotation=45, ha="right")

for ax in axs:
    ax.grid(True)

# legend
# axs[0].legend(ncol=2, fontsize="small", loc="upper center")

plt.tight_layout()


###############################
# %% scatter z_pred_L2 against z_pred_sample and z_pred_L2 against z_true
# in two hist2d plots

plt.figure(figsize=(12, 8))

y = results[0]["z_pred_L2"]

ax1 = plt.subplot(2, 1, 1)
x1 = results[0]["z_pred_sample1"] - y
h1 = ax1.hist2d(
    y,
    x1,
    bins=250,
    cmap="plasma",
    norm=matplotlib.colors.LogNorm(vmin=1, vmax=5000),
)
ax1.set_ylabel("FlexZBoost sample\n-\nFlexZBoost posterior mean")
ax1.set_xticklabels([])
ax1.grid(True)

ax2 = plt.subplot(2, 1, 2)
x2 = results[0]["z_true"] - y
h2 = ax2.hist2d(
    y,
    x2,
    bins=250,
    cmap="plasma",
    norm=matplotlib.colors.LogNorm(vmin=1, vmax=5000),
)
ax2.set_xlabel("FlexZBoost posterior mean")
ax2.set_ylabel("Truth\n-\nFlexZBoost posterior mean")
ax2.grid(True)

plt.tight_layout()
plt.show()

# %%


###############################
# %% scatter z_pred_L2 against z_pred_sample and z_pred_L2 against z_true
# for the normal boost model

plt.figure(figsize=(12, 8))

y = result_normal_boost["z_pred_L2"]

ax1 = plt.subplot(2, 1, 1)
x1 = result_normal_boost["z_pred_sample1"] - y
h1 = ax1.hist2d(
    y,
    x1,
    bins=250,
    cmap="plasma",
    norm=matplotlib.colors.LogNorm(vmin=1, vmax=5000),
)
ax1.set_ylabel("Gaussian sample\n-\nGaussian posterior mean")
ax1.set_xticklabels([])
ax1.grid(True)

ax2 = plt.subplot(2, 1, 2)
x2 = result_normal_boost["z_true"] - y
h2 = ax2.hist2d(
    y,
    x2,
    bins=250,
    cmap="plasma",
    norm=matplotlib.colors.LogNorm(vmin=1, vmax=5000),
)
ax2.set_xlabel("Gaussian posterior mean")
ax2.set_ylabel("Truth\n-\nGaussian posterior mean")
ax2.grid(True)

plt.tight_layout()
plt.show()
# %%
