# pylint: skip-file
from collections import defaultdict
from pathlib import Path
from typing import DefaultDict, Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from einops import rearrange, repeat
from hydra.utils import instantiate
from joblib import Parallel, delayed
from matplotlib import pyplot as plt
from torch import Tensor
from torch.distributions import Normal, Poisson
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
    if "test" in cfg.reconstruct:
        h = cfg.reconstruct.test.h
        w = cfg.reconstruct.test.w
        h_end = h + int(cfg.reconstruct.test.size)
        w_end = w + int(cfg.reconstruct.test.size)

    hlims = (h, h_end)
    wlims = (w, w_end)
    _, tile_map_recon = reconstruct_scene_at_coordinates(
        encoder,
        dec,
        frame.image,
        frame.background,
        hlims,
        wlims,
        slen=cfg.reconstruct.slen,
        device=device,
    )
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

    encoder_lower_threshold = Encoder(
        encoder.location_encoder,
        encoder.binary_encoder,
        encoder.galaxy_encoder,
        map_n_source_weights=cfg.reconstruct.map_n_source_weights,
    ).to(encoder.device)
    _, tile_map_lower_threshold = reconstruct_scene_at_coordinates(
        encoder_lower_threshold,
        dec,
        frame.image,
        frame.background,
        hlims,
        wlims,
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
        mismatch_dict = defaultdict(dict)
        detection_threshold = positive_negative_stats["true_matches"].float().mean(dim=0)

        photo_catalog = catalogs.get("photo")
        if photo_catalog is not None:
            row_indx, _, d, _ = reporting.match_by_locs(
                true_cat.plocs[0], photo_catalog.plocs[0], 1.0
            )
            photo_true_matches = torch.zeros(true_cat.plocs.shape[1], dtype=torch.bool)
            photo_true_matches[row_indx] = d
        else:
            photo_true_matches = None

        for i, ploc in enumerate(true_cat.plocs[0]):
            if bright_mismatches[i]:
                h = max(int(ploc[0].item() - 100.0), 0) + 24
                w = max(int(ploc[1].item() - 100.0), 0) + 24
                size = 200
                fig = create_figure_at_point(
                    h, w, size, bp, tile_map_recon, frame, dec, est_catalog=true_cat
                )
                filename = f"h{int(h)}_w{int(w)}.png"
                fig.savefig(mismatch_dir / filename)
                mismatch_dict["filename"][i] = filename
                mismatch_dict["h"][i] = ploc[0].item() + 24
                mismatch_dict["w"][i] = ploc[1].item() + 24
                mismatch_dict["ra"][i] = (
                    true_cat["ra"][0, i, 0].item() if "ra" in true_cat else None
                )
                mismatch_dict["dec"][i] = (
                    true_cat["dec"][0, i, 0].item() if "dec" in true_cat else None
                )
                mismatch_dict["mag"][i] = true_cat["mags"][0, i, 0].item()
                mismatch_dict["galaxy_bool"][i] = true_cat["galaxy_bools"][0, i, 0].item()
                mismatch_dict["detection_threshold"][i] = detection_threshold[i].item()
                if photo_true_matches is not None:
                    mismatch_dict["matched_by_photo"][i] = photo_true_matches[i].item()
        mismatch_tbl = pd.DataFrame(mismatch_dict)
        mismatch_tbl.sort_values("filename").to_csv(mismatch_dir / "mismatches.csv")


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
    encoder = Encoder(location.eval(), binary.eval(), galaxy.eval()).to(device)

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


def calc_scene_metrics_by_mag(
    est_cat: FullCatalog, true_cat: FullCatalog, mag_start: int, mag_end: int, loc_slack: float
):
    scene_metrics_by_mag: Dict[str, Dict[str, Number]] = {}
    mag_mins = [float(m - 1) for m in range(mag_start, mag_end + 1)] + [-np.inf]
    mag_maxes = [float(m) for m in range(mag_start, mag_end + 1)] + [mag_end]
    for mag_min, mag_max in zip(mag_mins, mag_maxes):
        # report counts on each bin
        true_cat_binned = true_cat.apply_mag_bin(mag_min, mag_max)
        est_cat_binned = est_cat.apply_mag_bin(mag_min, mag_max)
        tcount = true_cat_binned.n_sources.int().item()
        tgcount = true_cat_binned["galaxy_bools"].sum().int().item()
        tscount = tcount - tgcount

        detection_metrics = reporting.DetectionMetrics(loc_slack)
        classification_metrics = reporting.ClassificationMetrics(loc_slack)

        # precision
        est_cat_binned = est_cat.apply_mag_bin(mag_min, mag_max)
        detection_metrics.update(true_cat, est_cat_binned)
        precision_metrics = detection_metrics.compute()
        fp = precision_metrics["fp"].item()
        # reset global state since recall and precision use different cuts.
        detection_metrics.reset()

        # recall
        true_cat_binned = true_cat.apply_mag_bin(mag_min, mag_max)
        detection_metrics.update(true_cat_binned, est_cat)
        detection_dict = detection_metrics.compute()
        tp = detection_dict["tp"].item()
        tp_gal = detection_dict["n_galaxies_detected"].item()
        tp_star = tp - tp_gal
        detection_metrics.reset()

        # classification
        classification_metrics.update(true_cat_binned, est_cat)
        classification_result = classification_metrics.compute()
        n_matches = classification_result["n_matches"].item()

        conf_matrix = classification_result["conf_matrix"]
        galaxy_acc = conf_matrix[0, 0] / (conf_matrix[0, 0] + conf_matrix[0, 1])
        star_acc = conf_matrix[1, 1] / (conf_matrix[1, 1] + conf_matrix[1, 0])

        if np.isinf(mag_min):
            mag = "overall"
        else:
            mag = str(int(mag_max))

        scene_metrics_by_mag[mag] = {
            "tcount": tcount,
            "tgcount": tgcount,
            "tp": tp,
            "fp": fp,
            "recall": tp / tcount if tcount > 0 else 0.0,
            "precision": tp / (tp + fp) if (tp + fp) > 0 else 1.0,
            "tp_gal": tp_gal,
            "recall_gal": tp_gal / tgcount if tgcount > 0 else 0.0,
            "tp_star": tp_star,
            "recall_star": tp_star / tscount if tscount > 0 else 0.0,
            "classif_n_matches": n_matches,
            "classif_acc": classification_result["class_acc"].item(),
            "classif_galaxy_acc": galaxy_acc.item(),
            "classif_star_acc": star_acc.item(),
        }

    d: DefaultDict[str, Dict[str, Number]] = defaultdict(dict)
    for mag, scene_metrics_mag in scene_metrics_by_mag.items():
        for measure, value in scene_metrics_mag.items():
            d[measure][mag] = value
    return pd.DataFrame(d)


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
    if h + size + bp > frame.image.shape[2]:
        h = frame.image.shape[2] - size - bp
    if w + size + bp > frame.image.shape[3]:
        w = frame.image.shape[3] - size - bp
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

        mismatched = coadd_objects.get("mismatched")
        if mismatched is not None:
            locs_mismatched_true = locs_true[mismatched.squeeze(-1) > 0.5]
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
        bbox_to_anchor=(0.0, -0.1, 1.0, 0.5),
        loc="lower left",
        ncol=4,
        mode="expand",
        borderaxespad=0.0,
    )
    plt.subplots_adjust(hspace=-0.4)
    plt.tight_layout()

    return fig


def expected_recall_for_threshold(tile_map: TileCatalog, threshold: float):
    prob_on = rearrange(tile_map["n_source_log_probs"], "n nth ntw 1 1 -> n nth ntw").exp()
    is_on_array = prob_on >= threshold
    prob_detected = prob_on * is_on_array
    prob_not_detected = prob_on * (~is_on_array)
    recall = prob_detected.sum() / (prob_detected.sum() + prob_not_detected.sum())
    return recall.item()


def expected_precision_for_threshold(tile_map: TileCatalog, threshold: float):
    prob_on = rearrange(tile_map["n_source_log_probs"], "n nth ntw 1 1 -> n nth ntw").exp()
    is_on_array = prob_on >= threshold
    if is_on_array.sum() == 0:
        return 1.0
    prob_detected = prob_on * is_on_array
    precision = prob_detected.sum() / is_on_array.sum()
    return precision.item()


def expected_positives_and_negatives(tile_map: TileCatalog, threshold: float) -> Dict[str, float]:
    prob_on: Tensor = rearrange(
        tile_map["n_source_log_probs"], "n nth ntw 1 1 -> (n nth ntw)"
    ).exp()
    is_on_array = prob_on >= threshold
    prob_detected = prob_on[is_on_array]
    prob_not_detected = prob_on[~is_on_array]
    tp = float(prob_detected.sum().item())
    fp = float((1 - prob_detected).sum().item()) if is_on_array.sum() > 0 else 0.0
    tn = float((1 - prob_not_detected).sum().item()) if (~is_on_array).sum() > 0 else 0.0
    fn = float(prob_not_detected.sum().item())
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn, "n_selected": float(is_on_array.sum())}


def expected_positives_plot(
    tile_map: TileCatalog, actual_results: Dict, map_n_source_weights: Tuple[float, float]
):
    base_size = 8
    figsize = (4 * base_size, 2 * base_size)
    fig, axes = plt.subplots(nrows=4, ncols=2, figsize=figsize)
    thresholds = np.linspace(0.01, 0.99, 99)
    expected_results_lists = defaultdict(list)
    for threshold in thresholds:
        res_at_threshold = expected_positives_and_negatives(tile_map, threshold)
        for measure, value_at_threshold in res_at_threshold.items():
            expected_results_lists[measure].append(value_at_threshold)
    expected_results: Dict[str, np.ndarray] = {}
    for measure, values in expected_results_lists.items():
        expected_results[measure] = np.array(values)
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


def get_detection_stats_for_thresholds(thresholds, pred, true):
    pred_prec = np.ones_like(thresholds)
    selected = pred["n_selected"] > 0
    pred_prec[selected] = pred["tp"][selected] / pred["n_selected"][selected]

    pred_rec = np.zeros_like(thresholds)
    selected = (pred["tp"] + pred["fn"]) > 0
    pred_rec[selected] = pred["tp"][selected] / (pred["tp"] + pred["fn"])[selected]

    true_prec = np.ones_like(thresholds)
    selected = pred["n_selected"] > 0
    true_prec[selected] = true["tp"][selected] / pred["n_selected"][selected]

    if true["n_obj"] > 0:
        true_rec = true["tp"] / true["n_obj"]
    else:
        true_rec = np.zeros_like(thresholds)

    stats_dict = {
        "thresholds": thresholds,
        "expected_precision": pred_prec,
        "expected_recall": pred_rec,
        "actual_precision": true_prec,
        "actual_recall": true_rec,
    }
    stats_dict.update({f"expected_{k}": v for k, v in pred.items()})
    stats_dict.update({f"actual_{k}": v for k, v in true.items() if k != "true_matches"})
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


def get_positive_negative_stats(
    true_cat: FullCatalog,
    est_tile_cat: TileCatalog,
    mag_max: float = np.inf,
):
    true_cat = true_cat.apply_mag_bin(-np.inf, mag_max)
    thresholds = np.linspace(0.01, 0.99, 99)
    log_probs = rearrange(est_tile_cat["n_source_log_probs"], "n nth ntw 1 1 -> n nth ntw")
    est_tile_cat = est_tile_cat.copy()

    res = Parallel(n_jobs=10)(
        delayed(stats_for_threshold)(true_cat.plocs, est_tile_cat, t, log_probs)
        for t in tqdm(thresholds)
    )
    out: Dict[str, Union[int, Tensor]] = {}
    for k in res[0]:
        out[k] = torch.stack([r[k] for r in res])
    out["n_obj"] = true_cat.plocs.shape[1]
    return out


def stats_for_threshold(
    true_plocs: Tensor, est_tile_cat: TileCatalog, threshold: float, log_probs: Tensor
):
    est_tile_cat.n_sources = log_probs >= np.log(threshold)
    est_cat = est_tile_cat.to_full_params()
    number_true = true_plocs.shape[1]
    number_est = est_cat.plocs.shape[1]
    true_matches = torch.zeros(true_plocs.shape[1], dtype=torch.bool)
    if number_true == 0 or number_est == 0:
        return {
            "tp": torch.tensor(0.0),
            "fp": torch.tensor(float(number_est)),
            "true_matches": true_matches,
        }
    row_indx, col_indx, d, _ = reporting.match_by_locs(true_plocs[0], est_cat.plocs[0], 1.0)
    true_matches[row_indx] = d
    tp = d.sum()
    fp = number_est - tp
    return {"tp": tp, "fp": fp, "true_matches": true_matches}


if __name__ == "__main__":
    pass
