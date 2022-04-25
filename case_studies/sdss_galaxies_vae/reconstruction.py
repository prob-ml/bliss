# flake8: noqa
# pylint: skip-file
import json
from collections import defaultdict
from pathlib import Path
from queue import Full
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
from torch.types import Number
from tqdm import tqdm

from bliss import reporting
from bliss.catalog import FullCatalog, TileCatalog
from bliss.datasets.sdss import PhotoFullCatalog, SloanDigitalSkySurvey, convert_flux_to_mag
from bliss.encoder import Encoder
from bliss.inference import (
    SDSSFrame,
    SemiSyntheticFrame,
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

Frame = Union[SDSSFrame, SimulatedFrame, SemiSyntheticFrame]


def reconstruct(cfg):
    if cfg.reconstruct.outdir is not None:
        outdir = Path(cfg.reconstruct.outdir)
        outdir.mkdir(exist_ok=True)
    else:
        outdir = None
    frame: Frame = instantiate(cfg.reconstruct.frame)
    device = torch.device(cfg.reconstruct.device)
    dec, encoder, prior = load_models(cfg, device)
    if cfg.reconstruct.photo_catalog is not None:
        photo_catalog = PhotoFullCatalog.from_file(**cfg.reconstruct.photo_catalog)
    else:
        photo_catalog = None

    bp = encoder.border_padding
    h = bp
    w = bp
    h_end = ((frame.image.shape[2] - 2 * bp) // 4) * 4 + bp
    w_end = ((frame.image.shape[3] - 2 * bp) // 4) * 4 + bp

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
    true = frame.image[:, :, h:h_end, w:w_end]
    resid = (true - recon) / recon.sqrt()

    tile_map_recon["galaxy_blends"] = infer_blends(tile_map_recon, 2)
    print(f"{(tile_map_recon['galaxy_blends'] > 1).sum()} galaxies are part of blends in image.")
    tile_map_recon["fluxes"] = (
        tile_map_recon["galaxy_bools"] * tile_map_recon["galaxy_fluxes"]
        + tile_map_recon["star_bools"] * tile_map_recon["fluxes"]
    )
    tile_map_recon["mags"] = torch.zeros_like(tile_map_recon["fluxes"])
    tile_map_recon["mags"][tile_map_recon.is_on_array > 0] = convert_flux_to_mag(
        tile_map_recon["fluxes"][tile_map_recon.is_on_array > 0]
    )

    full_map_recon = tile_map_recon.to_full_params()
    scene_metrics_by_mag: Dict[str, pd.DataFrame] = {}
    ground_truth_catalog = frame.get_catalog((h, h_end), (w, w_end))
    catalogs = {"bliss": full_map_recon}
    if photo_catalog is not None:
        photo_catalog_at_hw = photo_catalog.crop_at_coords(h, h_end, w, w_end)
        catalogs["photo"] = photo_catalog_at_hw
    for catalog_name, catalog in catalogs.items():
        scene_metrics_by_mag[catalog_name] = calc_scene_metrics_by_mag(
            catalog,
            ground_truth_catalog,
            cfg.reconstruct.mag_min,
            cfg.reconstruct.mag_max,
            loc_slack=1.0,
        )

    encoder.map_n_source_weights = torch.tensor(cfg.reconstruct.map_n_source_weights)
    _, tile_map_lower_threshold = reconstruct_scene_at_coordinates(
        encoder,
        dec,
        frame.image,
        frame.background,
        (h, h_end),
        (w, w_end),
        slen=cfg.reconstruct.slen,
        device=device,
    )
    positive_negative_stats = get_positive_negative_stats(
        ground_truth_catalog, tile_map_lower_threshold, mag_max=cfg.reconstruct.mag_max
    )
    fig_exp_precision, detection_stats = expected_positives_plot(
        tile_map_recon,
        positive_negative_stats,
        cfg.reconstruct.map_n_source_weights,
    )
    if outdir is not None:
        # Expected precision lpot
        # fig_exp_precision = expected_precision_plot(tile_map_recon, recalls, precisions)
        detection_stats.to_csv(outdir / "stats_by_threshold.csv")
        for catalog_name, scene_metrics in scene_metrics_by_mag.items():
            scene_metrics.to_csv(outdir / f"scene_metrics_{catalog_name}.csv")
        torch.save(ground_truth_catalog, outdir / "ground_truth_catalog.pt")
        torch.save(full_map_recon, outdir / "map_recon.pt")
        torch.save(tile_map_recon, outdir / "tile_map_recon.pt")

        scene_dir = outdir / "reconstructions" / "scenes"
        scene_dir.mkdir(parents=True, exist_ok=True)
        for scene_name, scene_locs in cfg.reconstruct.scenes.items():
            h: int = scene_locs["h"]
            w: int = scene_locs["w"]
            size: int = scene_locs["size"]
            fig = create_figure_at_point(h, w, size, bp, tile_map_recon, frame, dec)
            fig.savefig(scene_dir / f"{scene_name}.png")

        mismatch_dir = outdir / "reconstructions" / "mismatches"
        mismatch_dir.mkdir(exist_ok=True)
        mismatches_at_map = positive_negative_stats["true_matches"][49] == 0
        true_cat = ground_truth_catalog.apply_mag_bin(-np.inf, cfg.reconstruct.mag_max)
        bright_truths = true_cat["mags"][0, :, 0] <= 20.0

        bright_mismatches = mismatches_at_map & bright_truths
        true_cat.allowed_params = true_cat.allowed_params.union({"mismatched"})
        true_cat["mismatched"] = bright_mismatches.reshape(1, -1, 1)
        for i, ploc in enumerate(true_cat.plocs[0]):
            if bright_mismatches[i]:
                h = max(int(ploc[0].item() - 100.0), 0) + 24
                w = max(int(ploc[1].item() - 100.0), 0) + 24
                size = 200
                fig = create_figure_at_point(
                    h, w, size, bp, tile_map_recon, frame, dec, est_catalog=true_cat
                )
                fig.savefig(mismatch_dir / f"h{int(h)}_w{int(w)}.png")

            # full_map_cropped = full_map_recon.crop(
            #     h - 24,
            #     full_map_recon.height - (h - 24 + size),
            #     w - 24,
            #     full_map_recon.width - (w - 24 + size),
            # )
        # fig_exp_precision.savefig(outdir / "auroc.png", format="png")
        # fig = create_figure(
        #     true[0, 0],
        #     recon[0, 0],
        #     resid[0, 0],
        #     # coadd_objects=coadd_data,
        #     map_recon=full_map_recon,
        #     include_residuals=False,
        #     colorbar=False,
        #     scatter_on_true=False,
        # )
        # fig.savefig(outdir / "recon.pdf", format="pdf")
        # fig.savefig(outdir / "recon.png", format="png")
        # fig_scatter_on_true = create_figure(
        #     true[0, 0],
        #     recon[0, 0],
        #     resid[0, 0],
        #     map_recon=full_map_recon,
        #     include_residuals=False,
        #     colorbar=False,
        #     scatter_on_true=True,
        # )
        # fig_with_tile_map = create_figure(
        #     true[0, 0],
        #     recon[0, 0],
        #     resid[0, 0],
        #     map_recon=full_map_recon,
        #     include_residuals=False,
        #     colorbar=False,
        #     scatter_on_true=True,
        #     tile_map=tile_map_recon,
        # )
        # fig_with_tile_map.savefig(outdir / "recon_with_tile_map.pdf", format="pdf")
        # fig_scatter_on_true.savefig(outdir / "recon_scatter_on_true.pdf", format="pdf")
        # fig_with_coadd = create_figure(
        #     true[0, 0],
        #     recon[0, 0],
        #     resid[0, 0],
        #     coadd_objects=ground_truth_catalog,
        #     map_recon=full_map_recon,
        #     include_residuals=False,
        #     colorbar=False,
        #     scatter_on_true=True,
        # )
        # fig_with_coadd.savefig(outdir / "recon_coadd.pdf", format="pdf")
        # fig_with_coadd.savefig(outdir / "recon_coadd.png", format="png")
        # tc = tile_map_recon.copy()
        # log_probs = rearrange(tc["n_source_log_probs"], "n nth ntw 1 1 -> n nth ntw")
        # tc.n_sources = log_probs >= np.log(0.15)
        # fc = tc.to_full_params()
        # fig_with_coadd_lower_thresh = create_figure(
        #     true[0, 0],
        #     recon[0, 0],
        #     resid[0, 0],
        #     coadd_objects=ground_truth_catalog,
        #     map_recon=fc,
        #     include_residuals=False,
        #     colorbar=False,
        #     scatter_on_true=True,
        # )
        # fig_with_coadd_lower_thresh.savefig(outdir / "recon_coadd_lower_thresh.png", format="png")


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
        location.eval(), binary.eval(), galaxy.eval(), cfg.reconstruct.map_n_source_weights
    ).to(device)

    prior: ImagePrior = instantiate(cfg.models.prior).to(device).eval()
    return dec, encoder, prior


def get_scene_boundaries(scene_coords, frame_height, frame_width, bp) -> Tuple[int, int, int, int]:
    h, w, scene_size = scene_coords["h"], scene_coords["w"], scene_coords["size"]
    if scene_size == "all":
        h = bp
        w = bp
        h_end = ((frame_height - 2 * bp) // 4) * 4 + bp
        w_end = ((frame_width - 2 * bp) // 4) * 4 + bp
    else:
        h_end = h + scene_size
        w_end = w + scene_size
    return h, h_end, w, w_end

    # scene_metrics_by_mag = {}
    # ground_truth_catalog = frame.get_catalog((h, h_end), (w, w_end))
    # catalogs = {"bliss": full_map_recon}
    # if photo_catalog is not None:
    #     photo_catalog_at_hw = photo_catalog.crop_at_coords(h, h_end, w, w_end)
    #     catalogs["photo"] = photo_catalog_at_hw
    # for catalog_name, catalog in catalogs.items():


def calc_scene_metrics_by_mag(
    est_cat: FullCatalog, true_cat: FullCatalog, mag_start: int, mag_end: int, loc_slack: float
):
    scene_metrics_by_mag: Dict[Union[int, str], Dict[str, Number]] = {}
    mags: List[Union[int, str]] = list(range(mag_start, mag_end + 1)) + ["overall"]
    for mag in mags:
        if mag != "overall":
            assert isinstance(mag, int)
            mag_min = float(mag) - 1.0
            mag_max = float(mag)
        else:
            mag_min = -np.inf
            mag_max = float(mag_end)

        # report counts on each bin
        true_cat_binned = true_cat.apply_mag_bin(mag_min, mag_max)
        est_cat_binned = est_cat.apply_mag_bin(mag_min, mag_max)
        tcount = true_cat_binned.n_sources.int().item()
        tgcount = true_cat_binned["galaxy_bools"].sum().int().item()
        tscount = tcount - tgcount

        ecount = est_cat_binned.n_sources.int().item()
        egcount = est_cat_binned["galaxy_bools"].sum().int().item()
        escount = ecount - egcount

        # scene_metrics_map = reporting.scene_metrics(true_cat, est_cat, mag_min=mag_min, mag_max=mag_max)
        detection_metrics = reporting.DetectionMetrics(loc_slack)
        classification_metrics = reporting.ClassificationMetrics(loc_slack)

        # precision
        est_cat_binned = est_cat.apply_mag_bin(mag_min, mag_max)
        detection_metrics.update(true_cat, est_cat_binned)
        # fp = float(detection_metrics.compute()["precision"].item())
        fp = detection_metrics.compute()["fp"].item()
        detection_metrics.reset()  # reset global state since recall and precision use different cuts.

        # recall
        true_cat_binned = true_cat.apply_mag_bin(mag_min, mag_max)
        detection_metrics.update(true_cat_binned, est_cat)
        detection_dict = detection_metrics.compute()
        tp = detection_dict["tp"].item()
        tp_gal = detection_dict["n_galaxies_detected"].item()
        fp_gal = est_cat["galaxy_bools"].sum().item() - tp_gal
        detection_metrics.reset()

        # classification
        classification_metrics.update(true_cat_binned, est_cat)
        classification_result = classification_metrics.compute()
        n_matches = classification_result["n_matches"].item()
        n_matches_gal_coadd = classification_result["n_matches_gal_coadd"]

        conf_matrix = classification_result["conf_matrix"]
        galaxy_acc = conf_matrix[0, 0] / (conf_matrix[0, 0] + conf_matrix[0, 1])
        star_acc = conf_matrix[1, 1] / (conf_matrix[1, 1] + conf_matrix[1, 0])

        scene_metrics_by_mag[mag] = {
            "tcount": tcount,
            "tgcount": tgcount,
            "tp": tp,
            "fp": fp,
            "recall": tp / tcount,
            "precision": tp / (tp + fp),
            "tp_gal": tp_gal,
            "tpr_gal": tp_gal / tgcount,
            "fp_gal": fp_gal,
            "fpr_gal": fp_gal / (fp_gal + tp_gal),
            "classif_n_matches": n_matches,
            "classif_acc": classification_result["class_acc"].item(),
            "classif_galaxy_acc": galaxy_acc.item(),
            "classif_star_acc": star_acc.item(),
        }

    d = defaultdict(dict)
    for mag, scene_metrics_mag in scene_metrics_by_mag.items():
        for measure, value in scene_metrics_mag.items():
            d[measure][mag] = value
    return pd.DataFrame(d)

    # if catalog_name == "bliss":
    #     scene_metrics_by_mag[catalog_name][mag].update(
    #         expected_accuracy(tile_map_recon, mag_min=mag_min, mag_max=mag_max)
    #     )
    #     if mag == "overall":
    #         scene_metrics_by_mag[catalog_name][mag]["expected_recall"] = expected_recall(
    #             tile_map_recon
    #         )
    #         scene_metrics_by_mag[catalog_name][mag][
    #             "expected_precision"
    #         ] = expected_precision(tile_map_recon)
    #         positive_negative_stats = get_positive_negative_stats(
    #             true_cat, tile_map_recon, mag_max=mag_max
    #         )
    # return positive_negative_stats


def create_figure_at_point(
    h: int,
    w: int,
    size: int,
    bp: int,
    tile_map_recon: TileCatalog,
    frame: Frame,
    dec: ImageDecoder,
    est_catalog: Optional[FullCatalog] = None,
):
    tile_slen = tile_map_recon.tile_slen

    h_tile = (h - bp) // tile_slen
    w_tile = (w - bp) // tile_slen
    n_tiles = size // tile_slen
    hlims = bp + (h_tile * tile_slen), bp + ((h_tile + n_tiles) * tile_slen)
    wlims = bp + (w_tile * tile_slen), bp + ((w_tile + n_tiles) * tile_slen)

    tile_map_cropped = tile_map_recon.crop((h_tile, h_tile + n_tiles), (w_tile, w_tile + n_tiles))
    full_map_cropped = tile_map_cropped.to_full_params()

    img_cropped = frame.image[0, 0, hlims[0] : hlims[1], wlims[0] : wlims[1]]
    bg_cropped = frame.background[0, 0, hlims[0] : hlims[1], wlims[0] : wlims[1]]
    with torch.no_grad():
        recon_cropped = dec.render_images(tile_map_cropped.to(dec.device))
        recon_cropped = recon_cropped.to("cpu")[0, 0, bp:-bp, bp:-bp] + bg_cropped
        resid_cropped = (img_cropped - recon_cropped) / recon_cropped.sqrt()

    if est_catalog is not None:
        tile_est_catalog = est_catalog.to_tile_params(
            tile_map_cropped.tile_slen, tile_map_cropped.max_sources
        )
        tile_est_catalog_cropped = tile_est_catalog.crop(
            (h_tile, h_tile + n_tiles), (w_tile, w_tile + n_tiles)
        )
        est_catalog_cropped = tile_est_catalog_cropped.to_full_params()
    else:
        est_catalog_cropped = None
    return create_figure(
        img_cropped,
        recon_cropped,
        resid_cropped,
        map_recon=full_map_cropped,
        coadd_objects=est_catalog_cropped,
    )


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

        if "mismatched" in coadd_objects:
            locs_mismatched_true = locs_true[coadd_objects["mismatched"].squeeze(-1) > 0.5]
        else:
            locs_mismatched_true = None
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
                color=true_gal_col,
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
                color=true_star_col,
                marker="+",
                s=scatter_size,
                label="SDSS Stars",
            )
        if locs_mismatched_true is not None:
            if scatter_on_true:
                ax_true.scatter(
                    locs_mismatched_true[:, 1],
                    locs_mismatched_true[:, 0],
                    color="orange",
                    marker="+",
                    s=scatter_size,
                    label="Unmatched",
                )
            ax_recon.scatter(
                locs_mismatched_true[:, 1],
                locs_mismatched_true[:, 0],
                color="orange",
                marker="+",
                s=scatter_size,
                label="Unmatched",
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
                color=pred_star_col,
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


def expected_positives_plot(
    tile_map: TileCatalog, actual_results: Dict, map_n_source_weights: Tuple[float, float]
):
    base_size = 8
    figsize = (4 * base_size, 2 * base_size)
    fig, axes = plt.subplots(nrows=4, ncols=2, figsize=figsize)
    thresholds = np.linspace(0.01, 0.99, 99)
    expected_results = defaultdict(list)
    for threshold in thresholds:
        res = expected_positives_and_negatives(tile_map, threshold)
        for k, v in res.items():
            expected_results[k].append(v)
    for k in expected_results:
        expected_results[k] = np.array(expected_results[k])
    min_viable_threshold = map_n_source_weights[0] / (
        map_n_source_weights[0] + map_n_source_weights[1]
    )
    axes[0, 0].plot(thresholds, expected_results["tp"])
    axes[0, 0].axvline(min_viable_threshold)
    axes[0, 0].set_xlabel("Threshold")
    axes[0, 0].set_ylabel("Expected True Positives")

    axes[0, 1].plot(thresholds, expected_results["fp"])
    axes[0, 1].axvline(min_viable_threshold)
    axes[0, 1].set_xlabel("Threshold")
    axes[0, 1].set_ylabel("Expected False Positives")

    min_viable_idx = np.argmin(np.abs(thresholds - min_viable_threshold))
    max_viable_fp = expected_results["fp"][min_viable_idx]
    axes[1, 0].plot(expected_results["fp"], expected_results["tp"])
    axes[1, 0].set_xlabel("Expected False Positives")
    axes[1, 0].axvline(max_viable_fp)
    axes[1, 0].set_ylabel("Expected True Positives")

    axes[2, 0].plot(thresholds, expected_results["tp"], label="Expected True Positives")
    axes[2, 0].plot(thresholds, actual_results["tp"], label="Actual True Positives")
    axes[2, 0].axvline(min_viable_threshold)
    axes[2, 0].set_xlabel("Threshold")
    axes[2, 0].axvline(min_viable_threshold)
    axes[2, 0].set_ylabel("Actual True Positives")
    axes[2, 0].legend()

    axes[2, 1].plot(thresholds, expected_results["fp"], label="Expected False Positives")
    axes[2, 1].plot(thresholds, actual_results["fp"], label="Actual False Positives")
    axes[2, 1].axvline(min_viable_threshold)
    axes[2, 1].set_xlabel("Threshold")
    axes[2, 1].set_ylabel("Actual False Positives")
    axes[2, 1].legend()

    axes[3, 0].plot(
        expected_results["fp"], expected_results["fp"], label="Expected False Positives"
    )
    axes[3, 0].plot(expected_results["fp"], actual_results["fp"], label="Actual False Positives")
    axes[3, 0].set_xlabel("Expected False Positives")
    axes[3, 0].axvline(max_viable_fp)
    axes[3, 0].set_ylabel("Actual False Positives")
    axes[3, 0].legend()

    axes[3, 1].plot(expected_results["fp"], expected_results["tp"], label="Expected True Positives")
    axes[3, 1].plot(expected_results["fp"], actual_results["tp"], label="Actual True Positives")
    axes[3, 1].axhline(actual_results["n_obj"])
    axes[3, 1].set_xlabel("Expected False Positives")
    axes[3, 1].axvline(max_viable_fp)
    axes[3, 1].set_ylabel("True Positives")
    axes[3, 1].legend()

    detection_stats = get_detection_stats_for_thresholds(
        thresholds, expected_results, actual_results
    )

    return fig, detection_stats


def get_detection_stats_for_thresholds(thresholds, expected_results, actual_results):
    expected_precision = expected_results["tp"] / expected_results["n_selected"]
    expected_recall = expected_results["tp"] / (expected_results["tp"] + expected_results["fn"])
    actual_precision = actual_results["tp"] / (actual_results["tp"] + actual_results["fp"])
    actual_recall = actual_results["tp"] / actual_results["n_obj"]
    stats_dict = {
        "thresholds": thresholds,
        "expected_precision": expected_precision,
        "expected_recall": expected_recall,
        "actual_precision": actual_precision,
        "actual_recall": actual_recall,
    }
    stats_dict.update({f"expected_{k}": v for k, v in expected_results.items()})
    stats_dict.update({f"actual_{k}": v for k, v in actual_results.items() if k != "true_matches"})
    return pd.DataFrame(stats_dict)


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
        number_true = true_cat.plocs.shape[1]
        number_est = est_cat.plocs.shape[1]
        if number_true == 0 or number_est == 0:
            return {"tp": 0.0, "fp": float(number_est)}
        row_indx, col_indx, d, _ = reporting.match_by_locs(true_cat.plocs[0], est_cat.plocs[0], 1.0)
        true_matches = torch.zeros(true_cat.plocs.shape[1], dtype=torch.bool)
        true_matches[row_indx] = d
        tp = d.sum()
        fp = number_est - tp
        return {"tp": tp, "fp": fp, "true_matches": true_matches}

    res = Parallel(n_jobs=10)(delayed(stats_for_threshold)(t) for t in tqdm(thresholds))
    out = {}
    for k in res[0]:
        out[k] = torch.stack([r[k] for r in res])
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
