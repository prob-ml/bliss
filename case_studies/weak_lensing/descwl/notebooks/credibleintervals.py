# pylint: disable=too-many-statements,protected-access
import os
import sys

import numpy as np
import pytorch_lightning as pl
import torch
import yaml
from hydra import compose, initialize
from hydra.utils import instantiate
from matplotlib import pyplot as plt
from pytorch_lightning.utilities import move_data_to_device
from scipy.stats import linregress

from bliss.global_env import GlobalEnv

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Navigate to the BLISS root directory
bliss_root = "/gpfs/accounts/regier_root/regier0/taodingr/bliss"
os.chdir(bliss_root)

# Add BLISS root to Python path
if bliss_root not in sys.path:
    sys.path.insert(0, bliss_root)


def _setup_config_and_encoder(ckpt):
    """Setup configuration and encoder."""
    with initialize(config_path="../", version_base=None):
        cfg = compose("config_descwl", {f"train.pretrained_weights={ckpt}"})

    seed = pl.seed_everything(cfg.train.seed)
    GlobalEnv.seed_in_this_program = seed

    data_source = instantiate(cfg.train.data_source)
    data_source.setup("test")
    test_dl = data_source.test_dataloader()

    encoder = instantiate(cfg.encoder).to(device)
    encoder_state_dict = torch.load(cfg.train.pretrained_weights, map_location=device)["state_dict"]
    encoder.load_state_dict(encoder_state_dict)
    encoder = encoder.eval()

    return test_dl, encoder


def _compute_credible_intervals(test_dl, encoder):
    """Compute credible intervals for shear predictions."""
    confidence_levels = torch.linspace(0.05, 0.95, steps=19)
    ci_quantiles = torch.distributions.Normal(0, 1).icdf(1 - (1 - confidence_levels) / 2).to(device)

    shear1_true = torch.zeros(len(test_dl), device="cpu")
    shear2_true = torch.zeros(len(test_dl), device="cpu")
    shear1_ci_lower = torch.zeros(len(test_dl), len(ci_quantiles), device="cpu")
    shear1_ci_upper = torch.zeros(len(test_dl), len(ci_quantiles), device="cpu")
    shear2_ci_lower = torch.zeros(len(test_dl), len(ci_quantiles), device="cpu")
    shear2_ci_upper = torch.zeros(len(test_dl), len(ci_quantiles), device="cpu")

    for i, b in enumerate(test_dl):
        batch = move_data_to_device(b, device)
        with torch.no_grad():
            shear1_true[i] = batch["tile_catalog"]["shear_1"].squeeze().flatten()
            shear2_true[i] = batch["tile_catalog"]["shear_2"].squeeze().flatten()

            input_lst = [inorm.get_input_tensor(batch) for inorm in encoder.image_normalizers]
            inputs = torch.cat(input_lst, dim=2)
            x_cat_marginal = encoder.net(inputs).squeeze()

            shear1_ci_lower[i] = x_cat_marginal[0] - ci_quantiles * x_cat_marginal[1].exp().sqrt()
            shear1_ci_upper[i] = x_cat_marginal[0] + ci_quantiles * x_cat_marginal[1].exp().sqrt()
            shear2_ci_lower[i] = x_cat_marginal[2] - ci_quantiles * x_cat_marginal[3].exp().sqrt()
            shear2_ci_upper[i] = x_cat_marginal[2] + ci_quantiles * x_cat_marginal[3].exp().sqrt()

    return {
        "confidence_levels": confidence_levels,
        "shear1_true": shear1_true,
        "shear2_true": shear2_true,
        "shear1_ci_lower": shear1_ci_lower,
        "shear1_ci_upper": shear1_ci_upper,
        "shear2_ci_lower": shear2_ci_lower,
        "shear2_ci_upper": shear2_ci_upper,
    }


def _compute_coverage_probabilities(
    shear1_true,
    shear2_true,
    shear1_ci_lower,
    shear1_ci_upper,
    shear2_ci_lower,
    shear2_ci_upper,
):
    """Compute coverage probabilities."""
    shear1_coverage_probs = (
        (
            (shear1_ci_lower <= shear1_true.unsqueeze(-1))
            & (shear1_true.unsqueeze(-1) <= shear1_ci_upper)
        )
        .float()
        .mean(0)
    )
    shear2_coverage_probs = (
        (
            (shear2_ci_lower <= shear2_true.unsqueeze(-1))
            & (shear2_true.unsqueeze(-1) <= shear2_ci_upper)
        )
        .float()
        .mean(0)
    )
    return shear1_coverage_probs, shear2_coverage_probs


def _plot_coverage_probabilities(
    confidence_levels,
    shear1_coverage_probs,
    shear2_coverage_probs,
    setting,
):
    """Plot coverage probability plot."""
    fig, ax = plt.subplots(1, 2, figsize=(13, 6))
    fontsize = 20
    ticklabelsize = 16
    color = "darkorchid"
    s = 80

    shear_data = [
        (shear1_coverage_probs, r"$\gamma_1$"),
        (shear2_coverage_probs, r"$\gamma_2$"),
    ]
    for i, (coverage_probs, title) in enumerate(shear_data):
        ax[i].axline((0, 0), slope=1, linestyle="dotted", color="black", linewidth=2, zorder=0)
        ax[i].scatter(confidence_levels, coverage_probs, color=color, s=s, zorder=1)
        ax[i].set_title(title, fontsize=1.5 * fontsize)
        ax[i].set_xlabel("Nominal coverage probability", fontsize=fontsize)
        ax[i].set_ylabel("Empirical coverage probability", fontsize=fontsize)
        ax[i].tick_params(axis="both", which="major", labelsize=ticklabelsize)
        ax[i].set_xlim(0, 1)
        ax[i].set_ylim(0, 1)
        ax[i].spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    fig.savefig(
        f"/scratch/regier_root/regier0/taodingr/bliss/case_studies/weak_lensing/"
        f"descwl/notebooks/figures/{setting}_coverageprobs.png",
        dpi=300,
        transparent=True,
        bbox_inches="tight",
        pad_inches=0,
    )


def _plot_credible_intervals_for_shear(
    ax,
    shear_true,
    shear_ci_lower,
    shear_ci_upper,
    shear_coverage_probs,
    interval_idx,
    indexes,
    alpha,
    shear_label,
    axmin,
    axmax,
    fontsize,
    ticklabelsize,
):
    """Plot credible intervals for a single shear component."""
    ax.axline((0, 0), slope=1, linestyle="dotted", color="black", linewidth=2)
    shear_coverage = (
        (shear_ci_lower <= shear_true.unsqueeze(-1)) & (shear_true.unsqueeze(-1) <= shear_ci_upper)
    )[..., interval_idx]

    covered_legend = False
    uncovered_legend = False

    for index in indexes:
        if shear_coverage[index] and not covered_legend:
            covered_legend = True
            color_val = "green"
            label = f"covers ({round(100 * shear_coverage_probs[interval_idx].item(), 1)}%)"
        elif not shear_coverage[index] and not uncovered_legend:
            uncovered_legend = True
            color_val = "red"
            coverage_pct = round(100 * (1 - shear_coverage_probs[interval_idx].item()), 1)
            label = f"does not cover ({coverage_pct}%)"
        else:
            color_val = "green" if shear_coverage[index] else "red"
            label = None

        ax.vlines(
            x=shear_true[index],
            ymin=shear_ci_lower[..., interval_idx][index],
            ymax=shear_ci_upper[..., interval_idx][index],
            alpha=alpha,
            color=color_val,
            label=label,
        )

    ax.set_xlabel(shear_label, fontsize=fontsize)
    ax.set_ylabel(f"$\\widehat{{{shear_label[1:]}}}$", fontsize=fontsize)
    ax.tick_params(axis="both", which="major", labelsize=ticklabelsize)
    ax.legend(loc="upper left", prop={"size": ticklabelsize})
    ax.set_xlim(axmin, axmax)
    ax.set_ylim(axmin, axmax)
    ax.spines[["top", "right"]].set_visible(False)


def _plot_credible_intervals(
    shear1_true,
    shear2_true,
    shear1_ci_lower,
    shear1_ci_upper,
    shear2_ci_lower,
    shear2_ci_upper,
    shear1_coverage_probs,
    shear2_coverage_probs,
    test_dl,
    setting,
):
    """Plot credible interval plots."""
    fig, ax = plt.subplots(1, 2, figsize=(13, 6))
    fontsize = 20
    ticklabelsize = 16
    alpha = 0.5

    axmin = min(shear1_ci_lower.min(), shear2_ci_lower.min()).cpu() - 0.01
    axmax = max(shear1_ci_upper.max(), shear2_ci_upper.max()).cpu() + 0.01

    np.random.seed(0)
    indexes = np.arange(len(test_dl))
    interval_idx = 17

    _plot_credible_intervals_for_shear(
        ax[0],
        shear1_true,
        shear1_ci_lower,
        shear1_ci_upper,
        shear1_coverage_probs,
        interval_idx,
        indexes,
        alpha,
        r"$\gamma_1$",
        axmin,
        axmax,
        fontsize,
        ticklabelsize,
    )

    _plot_credible_intervals_for_shear(
        ax[1],
        shear2_true,
        shear2_ci_lower,
        shear2_ci_upper,
        shear2_coverage_probs,
        interval_idx,
        indexes,
        alpha,
        r"$\gamma_2$",
        axmin,
        axmax,
        fontsize,
        ticklabelsize,
    )

    fig.tight_layout()
    fig.savefig(
        f"/scratch/regier_root/regier0/taodingr/bliss/case_studies/weak_lensing/"
        f"descwl/notebooks/figures/{setting}_credibleintervals.png",
        dpi=300,
        transparent=True,
        bbox_inches="tight",
        pad_inches=0,
    )


def draw_credible_interval(ckpt, setting):
    """Draw credible interval plots for shear predictions."""
    test_dl, encoder = _setup_config_and_encoder(ckpt)

    intervals_data = _compute_credible_intervals(test_dl, encoder)
    confidence_levels = intervals_data["confidence_levels"]
    shear1_true = intervals_data["shear1_true"]
    shear2_true = intervals_data["shear2_true"]
    shear1_ci_lower = intervals_data["shear1_ci_lower"]
    shear1_ci_upper = intervals_data["shear1_ci_upper"]
    shear2_ci_lower = intervals_data["shear2_ci_lower"]
    shear2_ci_upper = intervals_data["shear2_ci_upper"]

    shear1_coverage_probs, shear2_coverage_probs = _compute_coverage_probabilities(
        shear1_true,
        shear2_true,
        shear1_ci_lower,
        shear1_ci_upper,
        shear2_ci_lower,
        shear2_ci_upper,
    )

    _plot_coverage_probabilities(
        confidence_levels, shear1_coverage_probs, shear2_coverage_probs, setting
    )

    _plot_credible_intervals(
        shear1_true,
        shear2_true,
        shear1_ci_lower,
        shear1_ci_upper,
        shear2_ci_lower,
        shear2_ci_upper,
        shear1_coverage_probs,
        shear2_coverage_probs,
        test_dl,
        setting,
    )


def draw_scatter(ckpt, setting):

    with initialize(config_path="../", version_base=None):
        cfg = compose(
            "config_descwl",
            {
                f"train.pretrained_weights={ckpt}",
            },
        )

    seed = pl.seed_everything(cfg.train.seed)
    GlobalEnv.seed_in_this_program = seed

    data_source = instantiate(cfg.train.data_source)
    data_source.setup("test")
    test_dl = data_source.test_dataloader()

    encoder = instantiate(cfg.encoder).to(device)
    encoder_state_dict = torch.load(cfg.train.pretrained_weights, map_location=device)["state_dict"]
    encoder.load_state_dict(encoder_state_dict)
    encoder = encoder.eval()

    shear1_true = torch.zeros(len(test_dl), device=device)
    shear1_pred = torch.zeros(len(test_dl), device=device)
    shear2_true = torch.zeros(len(test_dl), device=device)
    shear2_pred = torch.zeros(len(test_dl), device=device)
    test_loss = torch.zeros(len(test_dl), device=device)

    i = -1
    for b in test_dl:
        i += 1
        batch = move_data_to_device(b, device)

        shear1_true[i] = batch["tile_catalog"]["shear_1"].squeeze()
        shear2_true[i] = batch["tile_catalog"]["shear_2"].squeeze()

        with torch.no_grad():
            mode_cat = encoder.sample(batch, use_mode=True)
            test_loss[i] = encoder._compute_loss(batch, None)

        shear1_pred[i] = mode_cat["shear_1"].squeeze()
        shear2_pred[i] = mode_cat["shear_2"].squeeze()

    # MSE calculations available as ((shear1_true - shear1_pred) ** 2).mean()
    # and ((shear2_true - shear2_pred) ** 2).mean()

    # Correlation calculations available:
    # Pearson: np.corrcoef(shear1_true.flatten().cpu(), shear1_pred.flatten().cpu())[1, 0]
    # Spearman: spearmanr(shear1_true.flatten().cpu(), shear1_pred.flatten().cpu())[0]
    # Kendall: kendalltau(shear1_true.flatten().cpu(), shear1_pred.flatten().cpu())[0]

    # Test loss available as test_loss.mean()

    alpha = 0.5
    s = 50
    fontsize = 20
    ticklabelsize = 16
    bliss_color = "darkorchid"
    # baseline_color = "sienna"  # Unused variable
    axmin = (
        min(shear1_true.min(), shear2_true.min(), shear1_pred.min(), shear2_pred.min()).cpu()
        - 0.005
    )
    axmax = (
        max(shear1_true.max(), shear2_true.max(), shear1_pred.max(), shear2_pred.max()).cpu()
        + 0.005
    )

    fig, ax = plt.subplots(1, 2, figsize=(14, 6))

    lr1 = linregress(shear1_true.flatten().cpu().numpy(), shear1_pred.flatten().cpu().numpy())
    lr2 = linregress(shear2_true.flatten().cpu().numpy(), shear2_pred.flatten().cpu().numpy())

    # Shear 1
    ax[0].scatter(
        shear1_true.flatten().cpu().numpy(),
        shear1_pred.flatten().cpu().numpy(),
        color=bliss_color,
        alpha=alpha,
        s=s,
        zorder=1,
    )
    x_line = np.linspace(axmin, axmax, 100)
    y_line1 = lr1.slope * x_line + lr1.intercept
    ax[0].plot(x_line, y_line1, color="black", linewidth=2, alpha=0.4, zorder=2)

    ax[0].set_xlabel(r"True $\gamma_1$", fontsize=fontsize)
    ax[0].set_ylabel(r"$\widehat{\gamma}_1$ (posterior mean)", fontsize=fontsize)
    ax[0].tick_params(axis="both", which="major", labelsize=ticklabelsize)
    ax[0].set_xlim((axmin, axmax))
    ax[0].set_ylim((axmin, axmax))

    # Shear 2
    ax[1].scatter(
        shear2_true.flatten().cpu().numpy(),
        shear2_pred.flatten().cpu().numpy(),
        color=bliss_color,
        alpha=alpha,
        s=s,
        zorder=1,
    )
    y_line2 = lr2.slope * x_line + lr2.intercept
    ax[1].plot(x_line, y_line2, color="black", linewidth=2, alpha=0.4, zorder=2)

    ax[1].set_xlabel(r"True $\gamma_2$", fontsize=fontsize)
    ax[1].set_ylabel(r"$\widehat{\gamma}_2$ (posterior mean)", fontsize=fontsize)
    ax[1].tick_params(axis="both", which="major", labelsize=ticklabelsize)
    ax[1].set_xlim((axmin, axmax))
    ax[1].set_ylim((axmin, axmax))

    for a in ax.flat:
        a.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()

    fig.savefig(
        f"/scratch/regier_root/regier0/taodingr/bliss/case_studies/weak_lensing/"
        f"descwl/notebooks/figures/{setting}_scatterplots.png",
        dpi=300,
        transparent=True,
        bbox_inches="tight",
        pad_inches=0,
    )

    print(  # noqa: WPS421
        f"Shear 1:\nc ± 3SE = {lr1.intercept:.6f} ± "
        f"{3 * lr1.intercept_stderr:.6f}, m ± 3SE = {lr1.slope - 1:.6f} ± "
        f"{3 * lr1.stderr}\n"
    )
    print(  # noqa: WPS421
        f"Shear 2:\nc ± 3SE = {lr2.intercept:.6f} ± "
        f"{3 * lr2.intercept_stderr:.6f}, m ± 3SE = {lr2.slope - 1:.6f} ± "
        f"{3 * lr2.stderr}"
    )


def main():
    # Credible Interval generation started
    with open(
        "/scratch/regier_root/regier0/taodingr/bliss/case_studies/weak_lensing/"
        "descwl/notebooks/credibleinterval_config.yaml",
        "r",
        encoding="utf-8",
    ) as f:
        config = yaml.safe_load(f)

    draw_credible_interval(ckpt=config["ckpt"], setting=config["setting"])
    draw_scatter(ckpt=config["ckpt"], setting=config["setting"])
    # Simulation completed


if __name__ == "__main__":
    main()
