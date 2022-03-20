# flake8: noqa
# pylint: skip-file
from pathlib import Path
from typing import Tuple, Union

import numpy as np
import pandas as pd
import torch
from einops import rearrange, repeat
from hydra.utils import instantiate
from matplotlib import pyplot as plt
from torch.distributions import Normal

from bliss import reporting
from bliss.catalog import TileCatalog
from bliss.datasets.sdss import SloanDigitalSkySurvey, convert_flux_to_mag
from bliss.encoder import Encoder
from bliss.inference import (
    SDSSFrame,
    SimulatedFrame,
    infer_blends,
    reconstruct_scene_at_coordinates,
)
from bliss.models.decoder import ImageDecoder
from bliss.models.prior import ImagePrior
from case_studies.sdss_galaxies.plots import set_rc_params

# class Frame(ABC):
#     pass


def reconstruct(cfg):
    if cfg.reconstruct.outdir is not None:
        outdir = Path(cfg.reconstruct.outdir)
        outdir.mkdir(exist_ok=True)
    else:
        outdir = None
    frame: Union[SDSSFrame, SimulatedFrame] = instantiate(cfg.reconstruct.frame)
    device = torch.device(cfg.reconstruct.device)
    dec, encoder, prior = load_models(cfg, device)

    for scene_name, scene_coords in cfg.reconstruct.scenes.items():
        bp = encoder.border_padding
        h, w, scene_size = scene_coords["h"], scene_coords["w"], scene_coords["size"]
        if scene_size == "all":
            h = bp
            w = bp
            h_end = ((frame.image.shape[2] - 2 * bp) // 4) * 4 + bp
            w_end = ((frame.image.shape[3] - 2 * bp) // 4) * 4 + bp
        else:
            h_end = h + scene_size
            w_end = w + scene_size
        true = frame.image[:, :, h:h_end, w:w_end]
        recon, tile_map_recon = reconstruct_scene_at_coordinates(
            encoder,
            dec,
            frame.image,
            frame.background,
            (h, h_end),
            (w, w_end),
            slen=cfg.reconstruct.slen,
            device=device,
        )
        resid = (true - recon) / recon.sqrt()
        tile_map_recon["galaxy_blends"] = infer_blends(tile_map_recon, 2)
        print(
            f"{(tile_map_recon['galaxy_blends'] > 1).sum()} galaxies are part of blends in image."
        )
        map_recon = tile_map_recon.to_full_params()
        map_recon["fluxes"] = (
            map_recon["galaxy_bools"] * map_recon["galaxy_fluxes"]
            + map_recon["star_bools"] * map_recon["fluxes"]
        )
        map_recon["mags"] = convert_flux_to_mag(map_recon["fluxes"])
        # map_recon["plocs"] = map_recon["plocs"] - 0.5

        tile_map_recon["fluxes"] = (
            tile_map_recon["galaxy_bools"] * tile_map_recon["galaxy_fluxes"]
            + tile_map_recon["star_bools"] * tile_map_recon["fluxes"]
        )
        tile_map_recon["mags"] = convert_flux_to_mag(tile_map_recon["fluxes"])
        scene_metrics_by_mag = {}
        ground_truth_catalog = frame.get_catalog((h, h_end), (w, w_end))
        for mag in range(18, 25):
            scene_metrics_map = reporting.scene_metrics(
                ground_truth_catalog,
                map_recon,
                mag_cut=float(mag),
            )
            scene_metrics_by_mag[mag] = scene_metrics_map
            conf_matrix = scene_metrics_map["conf_matrix"]
            scene_metrics_by_mag[mag]["galaxy_accuracy"] = conf_matrix[0, 0] / (
                conf_matrix[0, 0] + conf_matrix[0, 1]
            )
            scene_metrics_by_mag[mag]["star_accuracy"] = conf_matrix[1, 1] / (
                conf_matrix[1, 1] + conf_matrix[1, 0]
            )
            scene_metrics_by_mag[mag].update(
                expected_accuracy(tile_map_recon, mag_cutoff=float(mag))
            )
            if mag == 24:
                scene_metrics_by_mag[mag]["expected_recall"] = expected_recall(tile_map_recon)
                scene_metrics_by_mag[mag]["expected_precision"] = expected_precision(tile_map_recon)
        if outdir is not None:
            fig = create_figure(
                true[0, 0],
                recon[0, 0],
                resid[0, 0],
                # coadd_objects=coadd_data,
                map_recon=map_recon,
                include_residuals=False,
                colorbar=False,
                scatter_on_true=False,
            )
            fig.savefig(outdir / (scene_name + ".pdf"), format="pdf")
            fig.savefig(outdir / (scene_name + ".png"), format="png")
            fig_scatter_on_true = create_figure(
                true[0, 0],
                recon[0, 0],
                resid[0, 0],
                map_recon=map_recon,
                include_residuals=False,
                colorbar=False,
                scatter_on_true=True,
            )
            fig_with_tile_map = create_figure(
                true[0, 0],
                recon[0, 0],
                resid[0, 0],
                map_recon=map_recon,
                include_residuals=False,
                colorbar=False,
                scatter_on_true=True,
                tile_map=tile_map_recon,
            )
            fig_with_tile_map.savefig(outdir / (scene_name + "_with_tile_map.pdf"), format="pdf")
            fig_scatter_on_true.savefig(
                outdir / (scene_name + "_scatter_on_true.pdf"), format="pdf"
            )
            fig_with_coadd = create_figure(
                true[0, 0],
                recon[0, 0],
                resid[0, 0],
                coadd_objects=ground_truth_catalog,
                map_recon=map_recon,
                include_residuals=False,
                colorbar=False,
                scatter_on_true=True,
            )
            fig_with_coadd.savefig(outdir / (scene_name + "_coadd.pdf"), format="pdf")
            # scene_metrics_table = create_scene_metrics_table(scene_coords)
            # scene_metrics_table.to_csv(outdir / (scene_name + "_scene_metrics_by_mag.csv"))
            torch.save(scene_metrics_by_mag, outdir / (scene_name + ".pt"))
            torch.save(ground_truth_catalog, outdir / (scene_name + "_ground_truth_catalog.pt"))
            torch.save(map_recon, outdir / (scene_name + "_map_recon.pt"))
            torch.save(tile_map_recon, outdir / (scene_name + "_tile_map_recon.pt"))


def get_sdss_data(sdss_dir, sdss_pixel_scale):
    run = 94
    camcol = 1
    field = 12
    bands = (2,)
    sdss_data = SloanDigitalSkySurvey(
        sdss_dir=sdss_dir,
        run=run,
        camcol=camcol,
        fields=(field,),
        bands=bands,
    )

    return {
        "image": sdss_data[0]["image"][0],
        "background": sdss_data[0]["background"][0],
        "wcs": sdss_data[0]["wcs"][0],
        "pixel_scale": sdss_pixel_scale,
    }


def load_models(cfg, device) -> Tuple[ImageDecoder, Encoder, ImagePrior]:
    sleep = instantiate(cfg.models.sleep).to(device).eval()
    sleep.load_state_dict(torch.load(cfg.predict.sleep_checkpoint, map_location=sleep.device))

    binary = instantiate(cfg.models.binary).to(device).eval()
    binary.load_state_dict(torch.load(cfg.predict.binary_checkpoint, map_location=binary.device))

    galaxy = instantiate(cfg.models.galaxy_encoder).to(device).eval()
    if cfg.reconstruct.real:
        galaxy_state_dict = torch.load(
            cfg.predict.galaxy_checkpoint_real, map_location=galaxy.device
        )
    else:
        galaxy_state_dict = torch.load(cfg.predict.galaxy_checkpoint, map_location=galaxy.device)
    galaxy.load_state_dict(galaxy_state_dict)
    location = sleep.image_encoder.to(device).eval()
    dec = sleep.image_decoder.to(device).eval()
    encoder = Encoder(location.eval(), binary.eval(), galaxy.eval()).to(device)

    prior = instantiate(cfg.models.prior).to(device).eval()
    return dec, encoder, prior


def create_figure(
    true,
    recon,
    res,
    coadd_objects=None,
    map_recon=None,
    include_residuals: bool = True,
    colorbar=True,
    scatter_size: int = 100,
    scatter_on_true: bool = True,
    tile_map=None,
):
    """Make figures related to detection and classification in SDSS."""
    plt.style.use("seaborn-colorblind")

    true_gal_col = "m"
    true_star_col = "b"
    pred_gal_col = "c"
    pred_star_col = "r"
    # true_gal_col = "#e78ac3"
    # true_star_col = "#8da0cb"
    # pred_gal_col = "#fbb4ae"`
    # pred_star_col = "#66c2a5"
    pad = 6.0
    set_rc_params(fontsize=22, tick_label_size="small", legend_fontsize="small")
    ncols = 2 + include_residuals
    figsize = (20 + 10 * include_residuals, 12)
    fig, axes = plt.subplots(nrows=1, ncols=ncols, figsize=figsize)
    assert len(true.shape) == len(recon.shape) == len(res.shape) == 2

    # pick standard ranges for residuals
    scene_size = max(true.shape[-2], true.shape[-1])

    ax_true = axes[0]
    ax_recon = axes[1]

    ax_true.set_title("Original Image", pad=pad)
    ax_recon.set_title("Reconstruction", pad=pad)

    # plot images
    reporting.plot_image(
        fig, ax_true, true, vrange=(800, 1200), colorbar=colorbar, cmap="gist_gray"
    )
    if not tile_map:
        reporting.plot_image(
            fig, ax_recon, recon, vrange=(800, 1200), colorbar=colorbar, cmap="gist_gray"
        )
    else:
        is_on_array = rearrange(tile_map["is_on_array"], "1 nth ntw 1 1 -> nth ntw 1 1")
        is_on_array = repeat(is_on_array, "nth ntw 1 1 -> nth ntw h w", h=4, w=4)
        is_on_array = rearrange(is_on_array, "nth ntw h w -> (nth h) (ntw w)")
        ax_recon.matshow(is_on_array, vmin=0, vmax=1, cmap="gist_gray")
        for grid in range(3, is_on_array.shape[-1], 4):
            ax_recon.axvline(grid + 0.5, color="purple")
            ax_recon.axhline(grid + 0.5, color="purple")

    if include_residuals:
        ax_res = axes[2]
        ax_res.set_title("Residual", pad=pad)
        vmin_res, vmax_res = -6.0, 6.0
        reporting.plot_image(fig, ax_res, res, vrange=(vmin_res, vmax_res))

    if coadd_objects is not None:
        locs_true = coadd_objects["plocs"]
        true_galaxy_bools = coadd_objects["galaxy_bools"]
        locs_galaxies_true = locs_true[true_galaxy_bools.squeeze(-1) > 0.5]
        locs_stars_true = locs_true[true_galaxy_bools.squeeze(-1) < 0.5]
        if locs_galaxies_true.shape[0] > 0:
            if scatter_on_true:
                ax_true.scatter(
                    locs_galaxies_true[:, 1],
                    locs_galaxies_true[:, 0],
                    color=true_gal_col,
                    marker="+",
                    s=scatter_size,
                    label="COADD Galaxies",
                )
            ax_recon.scatter(
                locs_galaxies_true[:, 1],
                locs_galaxies_true[:, 0],
                color="m",
                marker="+",
                s=scatter_size,
                label="SDSS Galaxies",
            )
        if locs_stars_true.shape[0] > 0:
            if scatter_on_true:
                ax_true.scatter(
                    locs_stars_true[:, 1],
                    locs_stars_true[:, 0],
                    color=true_star_col,
                    marker="+",
                    s=scatter_size,
                    label="COADD Stars",
                )
            ax_recon.scatter(
                locs_stars_true[:, 1],
                locs_stars_true[:, 0],
                color="b",
                marker="+",
                s=scatter_size,
                label="SDSS Stars",
            )
            if include_residuals:
                ax_res.scatter(
                    locs_galaxies_true[:, 1],
                    locs_galaxies_true[:, 0],
                    color="b",
                    marker="+",
                    s=scatter_size,
                )
            if include_residuals:
                ax_res.scatter(
                    locs_galaxies_true[:, 1],
                    locs_galaxies_true[:, 0],
                    color="m",
                    marker="+",
                    s=scatter_size,
                )

    if map_recon is not None:
        locs_pred = map_recon["plocs"][0]
        star_bools = map_recon["star_bools"][0]
        galaxy_bools = map_recon["galaxy_bools"][0]
        locs_galaxies = locs_pred[galaxy_bools[:, 0] > 0.5, :]
        locs_stars = locs_pred[star_bools[:, 0] > 0.5, :]
        if locs_galaxies.shape[0] > 0:
            in_bounds = torch.all((locs_galaxies > 0) & (locs_galaxies < scene_size), dim=-1)
            locs_galaxies = locs_galaxies[in_bounds]
            if scatter_on_true:
                ax_true.scatter(
                    locs_galaxies[:, 1],
                    locs_galaxies[:, 0],
                    color=pred_gal_col,
                    marker="x",
                    s=scatter_size,
                    label="Predicted Galaxy",
                )
            ax_recon.scatter(
                locs_galaxies[:, 1],
                locs_galaxies[:, 0],
                color=pred_gal_col,
                marker="x",
                s=scatter_size,
                label="Predicted Galaxy",
                alpha=0.6,
            )
            if include_residuals:
                ax_res.scatter(
                    locs_galaxies[:, 1], locs_galaxies[:, 0], color="c", marker="x", s=scatter_size
                )
        if locs_stars.shape[0] > 0:
            in_bounds = torch.all((locs_stars > 0) & (locs_stars < scene_size), dim=-1)
            locs_stars = locs_stars[in_bounds]
            if scatter_on_true:
                ax_true.scatter(
                    locs_stars[:, 1],
                    locs_stars[:, 0],
                    color=pred_star_col,
                    marker="x",
                    s=scatter_size,
                    alpha=0.6,
                    label="Predicted Star",
                )
            ax_recon.scatter(
                locs_stars[:, 1],
                locs_stars[:, 0],
                color="r",
                marker="x",
                s=scatter_size,
                label="Predicted Star",
                alpha=0.6,
            )
            if include_residuals:
                ax_res.scatter(
                    locs_stars[:, 1], locs_stars[:, 0], color="r", marker="x", s=scatter_size
                )

    ax_recon.legend(
        # bbox_to_anchor=(0.0, 0.0, 0.0, 0.0),
        bbox_to_anchor=(0.0, -0.1, 1.0, 0.5),
        loc="lower left",
        ncol=4,
        mode="expand",
        borderaxespad=0.0,
    )
    plt.subplots_adjust(hspace=-0.4)
    plt.tight_layout()

    return fig


def create_scene_accuracy_table(scene_metrics_by_mag):
    tex_lines = []
    for k, v in scene_metrics_by_mag.items():
        line = f"{k} & {v['class_acc'].item():.2f} ({v['expected_accuracy'].item():.2f}) & {v['galaxy_accuracy']:.2f} ({v['expected_galaxy_accuracy']:.2f}) & {v['star_accuracy']:.2f} ({v['expected_star_accuracy']:.2f})\\\\\n"
        tex_lines.append(line)
    return tex_lines


def create_scene_metrics_table(scene_metrics_by_mag):
    x = {}
    columns = ("recall", "expected_recall", "precision", "expected_precision")
    for c in columns:
        x[c] = {}
        for k, v in scene_metrics_by_mag.items():
            if c in v:
                x[c][k] = v[c].item()
    scene_metrics_df = pd.DataFrame(x)
    tex_lines = []
    for k, v in scene_metrics_df.iterrows():
        line = f"{k} & {v['recall']:.2f}"
        if not np.isnan(v["expected_recall"]):
            line += f" ({v['expected_recall']:.2f})"
        line += f" & {v['precision']:.2f}"
        if not np.isnan(v["expected_precision"]):
            line += f" ({v['expected_precision']:.2f})"
        line += " \\\\\n"
        tex_lines.append(line)
    return scene_metrics_df, tex_lines


def expected_recall(tile_map: TileCatalog):
    source_probs = tile_map.n_sources_log_prob.unsqueeze(-1).unsqueeze(-1).exp()
    prob_if_on = source_probs * tile_map.is_on_array.unsqueeze(-1)
    prob_if_off = (1 - source_probs) * (1 - tile_map.is_on_array.unsqueeze(-1))
    recall = prob_if_on.sum() / (prob_if_on.sum() + prob_if_off.sum())
    return recall


def expected_precision(tile_map: TileCatalog):
    source_probs = tile_map.n_sources_log_prob.unsqueeze(-1).unsqueeze(-1).exp()
    prob_if_on = source_probs * tile_map.is_on_array.unsqueeze(-1)
    precision = prob_if_on.sum() / tile_map.is_on_array.unsqueeze(-1).sum()
    return precision


def expected_accuracy(tile_map, mag_cutoff=24.0):
    gal_probs = (
        tile_map["galaxy_bools"] * tile_map["galaxy_probs"] * (tile_map["mags"] <= mag_cutoff)
    )
    star_probs = (
        tile_map["star_bools"] * (1 - tile_map["galaxy_probs"]) * (tile_map["mags"] <= mag_cutoff)
    )
    expected_accuracy = (gal_probs + star_probs).sum() / (
        (tile_map["galaxy_bools"] + tile_map["star_bools"]) * (tile_map["mags"] <= mag_cutoff)
    ).sum()
    expected_galaxy_accuracy = (
        gal_probs.sum() / (tile_map["galaxy_bools"] * (tile_map["mags"] <= mag_cutoff)).sum()
    )
    expected_star_accuracy = (
        star_probs.sum() / (tile_map["star_bools"] * (tile_map["mags"] <= mag_cutoff)).sum()
    )
    return {
        "expected_accuracy": expected_accuracy,
        "expected_galaxy_accuracy": expected_galaxy_accuracy,
        "expected_star_accuracy": expected_star_accuracy,
    }


from torch.distributions import Poisson


def tile_map_prior(prior: ImagePrior, tile_map):
    # Source probabilities
    dist_sources = Poisson(torch.tensor(prior.mean_sources))
    log_prob_no_source = dist_sources.log_prob(torch.tensor(0))
    log_prob_one_source = dist_sources.log_prob(torch.tensor(1))
    log_prob_source = (tile_map["n_sources"] == 0) * log_prob_no_source + (
        tile_map["n_sources"] == 1
    ) * log_prob_one_source

    # Binary probabilities
    galaxy_log_prob = torch.tensor(0.7).log()
    star_log_prob = torch.tensor(0.3).log()
    log_prob_binary = (
        galaxy_log_prob * tile_map["galaxy_bools"] + star_log_prob * tile_map["star_bools"]
    )

    # Galaxy probabiltiies
    gal_dist = Normal(0.0, 1.0)
    galaxy_probs = gal_dist.log_prob(tile_map["galaxy_params"]) * tile_map["galaxy_bools"]

    # prob_normalized =
    return log_prob_source.sum() + log_prob_binary.sum() + galaxy_probs.sum()


if __name__ == "__main__":
    pass
