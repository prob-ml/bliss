# flake8: noqa
# pylint: skip-file
from collections import defaultdict
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from einops import rearrange, reduce, repeat
from hydra.utils import instantiate
from joblib import Parallel, delayed
from matplotlib import pyplot as plt
from torch import Tensor
from torch.distributions import Normal
from tqdm import tqdm

from bliss import reporting
from bliss.catalog import FullCatalog, TileCatalog
from bliss.datasets.sdss import PhotoFullCatalog, SloanDigitalSkySurvey, convert_flux_to_mag
from bliss.encoder import Encoder
from bliss.inference import (
    SDSSFrame,
    SimulatedFrame,
    infer_blends,
    reconstruct_scene_at_coordinates,
)
from bliss.models.binary import BinaryEncoder
from bliss.models.decoder import ImageDecoder
from bliss.models.galaxy_encoder import GalaxyEncoder
from bliss.models.location_encoder import LocationEncoder
from bliss.models.prior import ImagePrior
from case_studies.sdss_galaxies.plots import set_rc_params


def reconstruct(cfg):
    if cfg.reconstruct.outdir is not None:
        outdir = Path(cfg.reconstruct.outdir)
        outdir.mkdir(exist_ok=True)
    else:
        outdir = None
    frame: Union[SDSSFrame, SimulatedFrame] = instantiate(cfg.reconstruct.frame)
    device = torch.device(cfg.reconstruct.device)
    dec, encoder, prior = load_models(cfg, device)
    if cfg.reconstruct.photo_catalog is not None:
        photo_catalog = PhotoFullCatalog.from_file(**cfg.reconstruct.photo_catalog)
    else:
        photo_catalog = None

    for scene_name, scene_coords in cfg.reconstruct.scenes.items():
        assert isinstance(scene_name, str)
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

        tile_map_recon["fluxes"] = (
            tile_map_recon["galaxy_bools"] * tile_map_recon["galaxy_fluxes"]
            + tile_map_recon["star_bools"] * tile_map_recon["fluxes"]
        )
        tile_map_recon["mags"] = convert_flux_to_mag(tile_map_recon["fluxes"])
        scene_metrics_by_mag = {}
        ground_truth_catalog = frame.get_catalog((h, h_end), (w, w_end))
        catalogs = {"bliss": map_recon}
        if photo_catalog is not None:
            photo_catalog_at_hw = photo_catalog.crop_at_coords(h, h_end, w, w_end)
            catalogs["photo"] = photo_catalog_at_hw
        for catalog_name, catalog in catalogs.items():
            scene_metrics_by_mag[catalog_name] = {}
            for mag in list(range(cfg.reconstruct.mag_min, cfg.reconstruct.mag_max + 1)) + [
                "overall"
            ]:
                if mag != "overall":
                    mag_min = float(mag) - 1.0
                    mag_max = float(mag)
                else:
                    mag_min = -np.inf
                    mag_max = float(cfg.reconstruct.mag_max)
                scene_metrics_map = reporting.scene_metrics(
                    ground_truth_catalog,
                    catalog,
                    mag_min=mag_min,
                    mag_max=mag_max,
                    mag_slack=1.0,
                    mag_slack_accuracy=1.0,
                )
                scene_metrics_by_mag[catalog_name][mag] = scene_metrics_map
                conf_matrix = scene_metrics_map["conf_matrix"]
                scene_metrics_by_mag[catalog_name][mag]["galaxy_accuracy"] = conf_matrix[0, 0] / (
                    conf_matrix[0, 0] + conf_matrix[0, 1]
                )
                scene_metrics_by_mag[catalog_name][mag]["star_accuracy"] = conf_matrix[1, 1] / (
                    conf_matrix[1, 1] + conf_matrix[1, 0]
                )

                if catalog_name == "bliss":
                    scene_metrics_by_mag[catalog_name][mag].update(
                        expected_accuracy(tile_map_recon, mag_min=mag_min, mag_max=mag_max)
                    )
                    if mag == "overall":
                        scene_metrics_by_mag[catalog_name][mag][
                            "expected_recall"
                        ] = expected_recall(tile_map_recon)
                        scene_metrics_by_mag[catalog_name][mag][
                            "expected_precision"
                        ] = expected_precision(tile_map_recon)
                        positive_negative_stats = get_positive_negative_stats(
                            ground_truth_catalog, tile_map_recon, mag_max=mag_max
                        )
        if outdir is not None:
            # Expected precision lpot
            # fig_exp_precision = expected_precision_plot(tile_map_recon, recalls, precisions)
            fig_exp_precision, target_stats = expected_positives_plot(
                tile_map_recon, positive_negative_stats
            )
            fig_exp_precision.savefig(outdir / (scene_name + "_auroc.png"), format="png")
            with (outdir / (scene_name + "_auroc_target.json")).open("w") as fp:
                json.dump(target_stats, fp)
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
            fig_with_coadd.savefig(outdir / (scene_name + "_coadd.png"), format="png")
            tc = tile_map_recon.copy()
            log_probs = rearrange(tc["n_source_log_probs"], "n nth ntw 1 1 -> n nth ntw")
            tc.n_sources = log_probs >= np.log(0.15)
            fc = tc.to_full_params()
            fig_with_coadd_lower_thresh = create_figure(
                true[0, 0],
                recon[0, 0],
                resid[0, 0],
                coadd_objects=ground_truth_catalog,
                map_recon=fc,
                include_residuals=False,
                colorbar=False,
                scatter_on_true=True,
            )
            fig_with_coadd_lower_thresh.savefig(
                outdir / (scene_name + "_coadd_lower_thresh.png"), format="png"
            )
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
    # sleep = instantiate(cfg.models.sleep).to(device).eval()
    # sleep.load_state_dict(torch.load(cfg.predict.sleep_checkpoint, map_location=sleep.device))
    location: LocationEncoder = instantiate(cfg.models.location_encoder).to(device).eval()
    location.load_state_dict(
        torch.load(cfg.predict.location_checkpoint, map_location=location.device)
    )

    binary: BinaryEncoder = instantiate(cfg.models.binary).to(device).eval()
    binary.load_state_dict(torch.load(cfg.predict.binary_checkpoint, map_location=binary.device))

    galaxy: GalaxyEncoder = instantiate(cfg.models.galaxy_encoder).to(device).eval()
    if cfg.reconstruct.real:
        galaxy_state_dict = torch.load(
            cfg.predict.galaxy_checkpoint_real, map_location=galaxy.device
        )
    else:
        galaxy_state_dict = torch.load(cfg.predict.galaxy_checkpoint, map_location=galaxy.device)
    galaxy.load_state_dict(galaxy_state_dict)
    dec: ImageDecoder = instantiate(cfg.models.decoder).to(device).eval()
    encoder = Encoder(
        location.eval(), binary.eval(), galaxy.eval(), cfg.reconstruct.eval_mean_detections
    ).to(device)

    prior: ImagePrior = instantiate(cfg.models.prior).to(device).eval()
    return dec, encoder, prior


def create_figure(
    true,
    recon,
    res,
    coadd_objects: Optional[FullCatalog] = None,
    map_recon: Optional[FullCatalog] = None,
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
        is_on_array = rearrange(tile_map.is_on_array, "1 nth ntw 1 -> nth ntw 1 1")
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
        locs_true = coadd_objects.plocs - 0.5
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
        locs_pred = map_recon.plocs[0] - 0.5
        star_bools = map_recon["star_bools"][0, :, 0] > 0.5
        galaxy_bools = map_recon["galaxy_bools"][0, :, 0] > 0.5
        locs_galaxies = locs_pred[galaxy_bools, :]
        locs_stars = locs_pred[star_bools, :]
        locs_extra = locs_pred[(~galaxy_bools) & (~star_bools), :]
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

        if locs_extra.shape[0] > 0.5:
            in_bounds = torch.all((locs_extra > 0) & (locs_extra < scene_size), dim=-1)
            locs_extra = locs_extra[in_bounds]
            if scatter_on_true:
                ax_true.scatter(
                    locs_extra[:, 1],
                    locs_extra[:, 0],
                    color="w",
                    marker="x",
                    s=scatter_size,
                    alpha=0.6,
                    label="Predicted Object (below 0.5)",
                )
            ax_recon.scatter(
                locs_extra[:, 1],
                locs_extra[:, 0],
                color="w",
                marker="x",
                s=scatter_size,
                label="Predicted Object (below 0.5)",
                alpha=0.6,
            )
            if include_residuals:
                ax_res.scatter(
                    locs_extra[:, 1], locs_extra[:, 0], color="w", marker="x", s=scatter_size
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


def print_metrics_to_file(outdir):
    outdir = Path(outdir)
    out = collect_metrics(outdir)
    with outdir.joinpath("results.txt").open("w") as fp:
        for folder, folder_results in out.items():
            fp.write(str(folder) + "\n")
            for catalog, catalog_results in folder_results.items():
                fp.write(catalog + "\n")
                for metric, metric_results in catalog_results.items():
                    fp.write(metric + "\n")
                    fp.write(str(metric_results) + "\n")


def collect_metrics(outdir):
    out = {}
    for run in Path(outdir).iterdir():
        if run.is_dir():
            results = torch.load(run / "sdss_recon_all.pt")
            out[run] = {}
            for catalog in results:
                out[run][catalog] = {
                    "detections": create_scene_metrics_table(results[catalog])[0],
                    "accuracy": create_scene_accuracy_table(results[catalog]),
                }
    return out


def create_scene_accuracy_table(scene_metrics_by_mag):
    # tex_lines = []
    df = defaultdict(dict)
    cols = (
        "class_acc",
        "expected_accuracy",
        "galaxy_accuracy",
        "expected_galaxy_accuracy",
        "star_accuracy",
        "expected_star_accuracy",
        "n",
        "n_matches",
        "n_galaxies",
    )
    for k, v in scene_metrics_by_mag.items():
        # line = f"{k} & {v['class_acc'].item():.2f} ({v['expected_accuracy'].item():.2f}) & {v['galaxy_accuracy']:.2f} ({v['expected_galaxy_accuracy']:.2f}) & {v['star_accuracy']:.2f} ({v['expected_star_accuracy']:.2f})\\\\\n"
        for metric in cols:
            if metric in v:
                df[metric][k] = v[metric] if not isinstance(v[metric], Tensor) else v[metric].item()
        # tex_lines.append(line)
    df = pd.DataFrame(df)
    # return df, tex_lines
    return df


def create_scene_metrics_table(scene_metrics_by_mag):
    x = {}
    columns = (
        "recall",
        "expected_recall",
        "precision",
        "expected_precision",
        "n",
        "n_galaxies",
        "n_galaxies_detected",
    )
    for c in columns:
        x[c] = {}
        for k, v in scene_metrics_by_mag.items():
            v["n"] = v["counts"]["tgcount"] + v["counts"]["tscount"]
            v["n_galaxies"] = v["counts"]["tgcount"]
            if c in v:
                x[c][k] = v[c] if not isinstance(v[c], Tensor) else v[c].item()
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
    prob_on = rearrange(tile_map["n_source_log_probs"], "n nth ntw 1 1 -> n nth ntw").exp()
    is_on_array = rearrange(tile_map.is_on_array, "n nth ntw 1 -> n nth ntw")
    prob_detected = prob_on * is_on_array
    prob_not_detected = prob_on * (1 - is_on_array)
    recall = prob_detected.sum() / (prob_detected.sum() + prob_not_detected.sum())
    return recall


def expected_recall_for_threshold(tile_map: TileCatalog, threshold: float):
    prob_on = rearrange(tile_map["n_source_log_probs"], "n nth ntw 1 1 -> n nth ntw").exp()
    is_on_array = prob_on >= threshold
    prob_detected = prob_on * is_on_array
    prob_not_detected = prob_on * (~is_on_array)
    recall = prob_detected.sum() / (prob_detected.sum() + prob_not_detected.sum())
    return recall.item()


def expected_precision(tile_map: TileCatalog):
    prob_on = rearrange(tile_map["n_source_log_probs"], "n nth ntw 1 1 -> n nth ntw").exp()
    is_on_array = rearrange(tile_map.is_on_array, "n nth ntw 1 -> n nth ntw")
    prob_detected = prob_on * is_on_array
    precision = prob_detected.sum() / is_on_array.sum()
    return precision


def expected_precision_for_threshold(tile_map: TileCatalog, threshold: float):
    prob_on = rearrange(tile_map["n_source_log_probs"], "n nth ntw 1 1 -> n nth ntw").exp()
    is_on_array = prob_on >= threshold
    if is_on_array.sum() == 0:
        return 1.0
    prob_detected = prob_on * is_on_array
    precision = prob_detected.sum() / is_on_array.sum()
    return precision.item()


def expected_true_positives_for_threshold(tile_map: TileCatalog, threshold: float):
    prob_on: Tensor = rearrange(tile_map["n_source_log_probs"], "n nth ntw 1 1 -> n nth ntw").exp()
    is_on_array = prob_on >= threshold
    prob_detected = prob_on * is_on_array
    return prob_detected.sum().item()


def expected_false_positives_for_threshold(tile_map: TileCatalog, threshold: float):
    prob_on: Tensor = rearrange(tile_map["n_source_log_probs"], "n nth ntw 1 1 -> n nth ntw").exp()
    is_on_array = prob_on >= threshold
    prob_detected = prob_on * is_on_array
    return (1.0 - prob_detected).sum().item()


def expected_positives_and_negatives(tile_map: TileCatalog, threshold: float) -> Dict[str, float]:
    prob_on: Tensor = rearrange(
        tile_map["n_source_log_probs"], "n nth ntw 1 1 -> (n nth ntw)"
    ).exp()
    is_on_array = prob_on >= threshold
    # if is_on_array.sum() == 0:
    #     return {
    #         "tp": 0.0,
    #         "fp": 0.0,
    #         "tn": 0.0,
    #         "fn":
    #     }
    prob_detected = prob_on[is_on_array]
    prob_not_detected = prob_on[~is_on_array]
    tp = float(prob_detected.sum().item())
    fp = float((1 - prob_detected).sum().item()) if is_on_array.sum() > 0 else 0.0
    tn = float((1 - prob_not_detected).sum().item()) if (~is_on_array).sum() > 0 else 0.0
    fn = float(prob_not_detected.sum().item())
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn, "n_selected": float(is_on_array.sum())}


def expected_precision_plot(tile_map: TileCatalog, true_recalls, true_precisions):
    base_size = 8
    figsize = (3 * base_size, 2 * base_size)
    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=figsize)
    thresholds = np.linspace(0.01, 0.99, 99)
    precisions = []
    recalls = []
    for threshold in thresholds:
        precision = expected_precision_for_threshold(tile_map, threshold)
        recall = expected_recall_for_threshold(tile_map, threshold)
        precisions.append(precision)
        recalls.append(recall)
    precisions = np.array(precisions)
    recalls = np.array(recalls)
    axes[0, 0].scatter(thresholds, precisions)
    axes[0, 0].set_xlabel("Threshold")
    axes[0, 0].set_ylabel("Expected Precision")
    axes[0, 1].scatter(thresholds, recalls)
    axes[0, 1].set_xlabel("Threshold")
    axes[0, 1].set_ylabel("Expected Recall")

    # colors = precisions == precisions[optimal_point]
    axes[1, 0].scatter(precisions, recalls)
    optimal_point = np.argmin(1 / precisions + 1 / recalls)
    x, y = precisions[optimal_point], recalls[optimal_point]
    axes[1, 0].scatter(x, y, color="yellow", marker="+")
    axes[1, 0].annotate(
        f"Expected precision: {x:.2f}\nExpected Recall {y:.2f}", (x, y), fontsize=12
    )
    axes[1, 0].set_xlabel("Expected Precision")
    axes[1, 0].set_ylabel("Expected Recall")
    # axes[1,0].xlabel("Expected Precision")
    # axes[1,0].ylabel("Expected Recall")
    axes[2, 0].scatter(precisions, true_precisions)
    x, y = precisions[optimal_point], true_precisions[optimal_point]
    axes[2, 0].scatter(x, y, color="yellow", marker="+")
    axes[2, 0].annotate(f"Expected precision: {x:.2f}\nTrue precision {y:.2f}", (x, y), fontsize=12)
    axes[2, 0].set_xlabel("Expected Precision")
    axes[2, 0].set_ylabel("Actual Precision")
    axes[2, 1].scatter(precisions, true_recalls)
    x, y = precisions[optimal_point], true_recalls[optimal_point]
    axes[2, 1].scatter(x, y, color="yellow", marker="+")
    axes[2, 1].annotate(f"Expected precision: {x:.2f}\nTrue recall {y:.2f}", (x, y), fontsize=12)
    axes[2, 1].set_xlabel("Expected Precision")
    axes[2, 1].set_ylabel("Actual Recall")
    return fig


def expected_positives_plot(tile_map: TileCatalog, actual_results: Dict[str, float]):
    base_size = 8
    figsize = (4 * base_size, 2 * base_size)
    fig, axes = plt.subplots(nrows=4, ncols=2, figsize=figsize)
    thresholds = np.linspace(0.01, 0.99, 99)
    results = defaultdict(list)
    for threshold in thresholds:
        res = expected_positives_and_negatives(tile_map, threshold)
        for k, v in res.items():
            results[k].append(v)
    for k in results:
        results[k] = np.array(results[k])
    axes[0, 0].plot(thresholds, results["tp"])
    axes[0, 0].set_xlabel("Threshold")
    axes[0, 0].set_ylabel("Expected True Positives")

    axes[0, 1].plot(thresholds, results["fp"])
    axes[0, 1].set_xlabel("Threshold")
    axes[0, 1].set_ylabel("Expected False Positives")

    axes[1, 0].plot(results["fp"], results["tp"])
    axes[1, 0].set_xlabel("Expected False Positives")
    axes[1, 0].set_ylabel("Expected True Positives")

    axes[2, 0].plot(thresholds, actual_results["tp"])
    axes[2, 0].set_xlabel("Threshold")
    axes[2, 0].set_ylabel("Actual True Positives")

    axes[2, 1].plot(thresholds, actual_results["fp"])
    axes[2, 1].set_xlabel("Threshold")
    axes[2, 1].set_ylabel("Actual False Positives")

    axes[3, 0].plot(results["fp"], actual_results["fp"])
    axes[3, 0].set_xlabel("Expected False Positives")
    axes[3, 0].set_ylabel("Actual False Positives")

    axes[3, 1].plot(results["fp"], results["tp"], label="Expected True Positives")
    axes[3, 1].plot(results["fp"], actual_results["tp"], label="Actual True Positives")
    axes[3, 1].axhline(actual_results["n_obj"])
    axes[3, 1].set_xlabel("Expected False Positives")
    axes[3, 1].set_ylabel("True Positives")
    axes[3, 1].legend()

    precision = results["tp"] / results["n_selected"]
    target_idx_baseline = 15
    target_idx_precision = int(np.power(precision - 0.7, 2).argmin())
    target_stats = {}
    for target_idx in (target_idx_baseline, target_idx_precision):
        threshold = thresholds[target_idx]
        target_precision = precision[target_idx]
        target_recall = (results["tp"] / (results["tp"] + results["fn"]))[target_idx]
        actual_precision = (actual_results["tp"] / (actual_results["tp"] + actual_results["fp"]))[
            target_idx
        ]
        actual_recall = (actual_results["tp"] / actual_results["n_obj"])[target_idx]
        target_stats[target_idx] = {
            "threshold": threshold,
            "target_precision": target_precision,
            "target_recall": target_recall,
            "actual_precision": actual_precision,
            "actual_recall": actual_recall,
        }

    return fig, target_stats


def expected_accuracy(tile_map, mag_min, mag_max):
    keep = tile_map["mags"] <= mag_max
    keep = keep & (tile_map["mags"] >= mag_min)
    gal_probs = tile_map["galaxy_bools"] * tile_map["galaxy_probs"] * keep
    star_probs = tile_map["star_bools"] * (1 - tile_map["galaxy_probs"]) * keep
    expected_accuracy = (gal_probs + star_probs).sum() / (
        (tile_map["galaxy_bools"] + tile_map["star_bools"]) * keep
    ).sum()
    expected_galaxy_accuracy = gal_probs.sum() / (tile_map["galaxy_bools"] * keep).sum()
    expected_star_accuracy = star_probs.sum() / (tile_map["star_bools"] * keep).sum()
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


def match_by_locs_closest_pairs(
    true_locs: Tensor, est_locs: Tensor, max_l_infty_dist: float
) -> List[Tuple[int, int]]:
    """Match true locations to estimated locations by closest pairs.

    Arguments:
        true_locs: Tensor of true locations (N_true x 2)
        est_locs: Tensor of estimated locations (N_est x 2)
        max_l_infty_dist: Maximim l-infinity distance allowed for matches.

    Returns:
        A list of tuples (i, j) indicating a match between the i-th row of true_locs
        and the j-th row of est_locs.
    """
    assert len(true_locs.shape) == len(est_locs.shape) == 2
    assert true_locs.shape[-1] == est_locs.shape[-1] == 2
    assert isinstance(true_locs, torch.Tensor) and isinstance(est_locs, torch.Tensor)

    locs1 = true_locs.view(-1, 2)
    locs2 = est_locs.view(-1, 2)

    # entry (i,j) is l1 distance between of ith loc in locs1 and the jth loc in locs2
    locs_abs_diff = (rearrange(locs1, "i j -> i 1 j") - rearrange(locs2, "i j -> 1 i j")).abs()
    locs_err = reduce(locs_abs_diff, "i j k -> i j", "sum")
    locs_err_inf = reduce(locs_abs_diff, "i j k -> i j", "max")
    allowed_match = locs_err_inf <= max_l_infty_dist
    disallowed_penalty = torch.zeros_like(allowed_match, dtype=torch.float)
    disallowed_penalty[~allowed_match] = np.inf
    locs_err += disallowed_penalty

    return match_closest_pairs(locs_err)


def match_closest_pairs(distances: Tensor) -> List[Tuple[int, int]]:
    """Match pairs by closest distance.

    Given a matrix of distances with rows and columns corresponding to two
    sets of objects, this function matches them in the following way:
    1) The pair (i, j) with the closest distance gets matched.
    2) i and j are removed from consideration.
    3) The next-closest pair of the remaining objects gets matched.
    This process repeats until all remaining distances are infinite.
    A distance of infinity indicates there is no edge between i and j,
    and hence a match can never be made.

    Arguments:
        distances: A matrix of distances. Must be non-negative.

    Returns:
        A list of (i, j) pairs corresponding to row-column matches in the
        input distance matrix.
    """
    pairs = []
    dist_flat = distances.flatten()
    best_pair = dist_flat.argmin().item()
    dist = dist_flat[best_pair]
    while dist < np.inf and (distances.shape[0] > 0) and (distances.shape[1] > 1):
        best_row = best_pair // distances.shape[1]
        best_col = best_pair % distances.shape[1]
        pairs.append((best_row, best_col))
        distances = np.delete(distances, best_row, 0)
        distances = np.delete(distances, best_col, 1)
        dist_flat = distances.flatten()
        best_pair = dist_flat.argmin().item()
        dist = dist_flat[best_pair]
    return pairs


def get_positive_negative_stats(
    true_cat: FullCatalog,
    est_tile_cat: TileCatalog,
    mag_max: float = np.inf,
):
    true_cat = true_cat.apply_mag_bin(-np.inf, mag_max)
    thresholds = np.linspace(0.01, 0.99, 99)
    log_probs = rearrange(est_tile_cat["n_source_log_probs"], "n nth ntw 1 1 -> n nth ntw")
    est_tile_cat = est_tile_cat.copy()

    def stats_for_threshold(threshold):
        est_tile_cat.n_sources = log_probs >= np.log(threshold)
        est_cat = est_tile_cat.to_full_params()
        number_predicted = est_cat.plocs.shape[1]
        if number_predicted == 0:
            return {"tp": 0.0, "fp": 0.0}
        _, _, d, _ = reporting.match_by_locs(true_cat.plocs[0], est_cat.plocs[0], 1.0)
        tp = d.sum()
        fp = number_predicted - tp
        return {"tp": tp, "fp": fp}

    res = Parallel(n_jobs=10)(delayed(stats_for_threshold)(t) for t in tqdm(thresholds))
    out = {}
    for k in res[0]:
        out[k] = np.array([r[k] for r in res])
    out["n_obj"] = true_cat.plocs.shape[1]
    return out


import math


def _adj_log_probs(log_probs: Tensor, log_train_probs: Tensor, log_test_probs: Tensor):
    assert torch.allclose(log_train_probs.exp().sum(), torch.tensor(1.0))
    assert torch.allclose(log_test_probs.exp().sum(), torch.tensor(1.0))
    log_adj_ratios = (log_test_probs - log_train_probs).reshape(1, 1, 1, 2)
    # log_adj_ratio = math.log(adj_ratio)
    log_1m_probs = torch.log1p(-torch.exp(log_probs))
    log_probs_all = torch.stack((log_1m_probs, log_probs), dim=-1)
    log_probs_all_adj = log_probs_all + log_adj_ratios
    log_probs_all_adj_norm = torch.log_softmax(log_probs_all_adj, dim=-1)
    log_probs_adj_norm = log_probs_all_adj_norm[:, :, :, 1]
    return log_probs_adj_norm


def adj_prob(prob, adj_ratio):
    return prob * adj_ratio / (prob * adj_ratio + (1 - prob) / adj_ratio)


if __name__ == "__main__":
    pass
