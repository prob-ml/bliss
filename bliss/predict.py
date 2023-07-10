from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch
from astropy.wcs import WCS
from bokeh.models import TabPanel, Tabs
from bokeh.plotting import figure, output_file, show
from hydra.utils import instantiate
from reproject import reproject_interp
from torch import Tensor

from bliss.catalog import FullCatalog
from bliss.surveys.decals import DarkEnergyCameraLegacySurvey as DECaLS
from bliss.surveys.decals import DecalsDownloader, DecalsFullCatalog
from bliss.surveys.sdss import SloanDigitalSkySurvey as SDSS


def prepare_image(x, device):
    x = torch.from_numpy(x).unsqueeze(0)
    x = x.to(device=device)
    # image dimensions must be a multiple of 16
    height = x.size(2) - (x.size(2) % 16)
    width = x.size(3) - (x.size(3) % 16)
    return x[:, :, :height, :width]


def crop_image(image, background, crop_params):
    """Crop the image (and background) to a subregion for prediction."""
    if not crop_params.do_crop:
        return image, background

    idx0 = crop_params.left_upper_corner[0]
    idx1 = crop_params.left_upper_corner[1]
    width = crop_params.width
    height = crop_params.height
    if ((idx0 + height) <= image.shape[2]) and ((idx1 + width) <= image.shape[3]):
        image = image[:, :, idx0 : idx0 + height, idx1 : idx1 + width]
        background = background[:, :, idx0 : idx0 + height, idx1 : idx1 + width]
    return image, background


def prepare_batch(images, backgrounds):
    batch = {"images": images, "background": backgrounds}
    batch["images"] = batch["images"].squeeze(0)
    batch["background"] = batch["background"].squeeze(0)
    return batch


def align(img, ref_wcs, ref_band):
    """Reproject images based on some reference WCS for pixel alignment."""
    reproj_d = {}
    footprint_d = {}

    orig_dim = img.shape

    # align with r-band WCS
    for bnd in range(img.shape[0]):
        inputs = (img[bnd], ref_wcs[bnd])
        reproj, footprint = reproject_interp(  # noqa: WPS317
            inputs, ref_wcs[ref_band], order="bicubic", shape_out=img[bnd].shape
        )
        reproj_d[bnd] = reproj
        footprint_d[bnd] = footprint

    # use footprints to handle NaNs from reprojection
    h, w = footprint_d[0].shape
    out_print = np.ones((h, w))
    for fp in footprint_d.values():
        out_print = out_print * fp  # noqa: WPS350

    out_print = np.expand_dims(out_print, axis=0)

    reproj_out = np.zeros((5, orig_dim[1], orig_dim[2]))

    for i in range(img.shape[0]):
        reproj_d[i] = np.multiply(reproj_d[i], out_print)
        cropped = reproj_d[i][0, : orig_dim[1], : orig_dim[2]]
        cropped[np.isnan(cropped)] = 0
        reproj_out[i] = cropped

    return reproj_out


def nelec_to_nmgy_for_catalog(est_cat, nelec_per_nmgy_per_band):
    fluxes_suffix = "_fluxes"
    # reshape nelec_per_nmgy_per_band to (1, {n_bands}) to broadcast
    nelec_per_nmgy_per_band = nelec_per_nmgy_per_band.reshape(1, -1)
    for key in est_cat.keys():
        if key.endswith(fluxes_suffix):
            est_cat[key] = torch.tensor(np.array(est_cat[key]) / nelec_per_nmgy_per_band)
        elif key == "galaxy_params":
            clone = est_cat[key].clone()
            clone[..., :5] = torch.tensor(
                np.array(clone[..., :5]) / nelec_per_nmgy_per_band
            )  # TODO: remove magic number 5 with {n_bands}
            est_cat[key] = clone
    return est_cat


def predict(cfg) -> Tuple[FullCatalog, Tensor, Tensor, Any, Dict[str, Tensor]]:
    survey = instantiate(cfg.predict.dataset)
    assert (
        len(survey) == 1
    ), "Prediction only supported for one image_id (e.g., (run, camcol, field))"
    survey_obj = survey[0]
    survey_obj["image"] = align(
        survey_obj["image"], ref_wcs=survey_obj["wcs"], ref_band=cfg.predict.dataset.reference_band
    )
    survey_obj["background"] = align(
        survey_obj["background"],
        ref_wcs=survey_obj["wcs"],
        ref_band=cfg.predict.dataset.reference_band,
    )

    cat_path = survey.downloader.download_catalog()
    plocs = survey.catalog_cls.from_file(
        cat_path=cat_path,
        wcs=survey_obj["wcs"][cfg.predict.dataset.reference_band],
        height=survey_obj["image"].shape[1],
        width=survey_obj["image"].shape[2],
    ).plocs[0]

    encoder = instantiate(cfg.encoder).to(cfg.predict.device)
    enc_state_dict = torch.load(cfg.predict.weight_save_path)
    encoder.load_state_dict(enc_state_dict)
    encoder.eval()
    trainer = instantiate(cfg.predict.trainer)
    images = prepare_image(survey.predict_batch["images"], cfg.predict.device)
    backgrounds = prepare_image(survey.predict_batch["background"], cfg.predict.device)
    images, backgrounds = crop_image(images, backgrounds, cfg.predict.crop)
    survey.predict_batch = prepare_batch(images, backgrounds)
    est_cat, images, background, pred = trainer.predict(encoder, datamodule=survey)[0].values()

    # mean of the nelec_per_mgy per band
    nelec_per_nmgy_per_band = np.mean(survey_obj["nelec_per_nmgy_list"], axis=1)
    est_cat = nelec_to_nmgy_for_catalog(est_cat, nelec_per_nmgy_per_band)

    est_full = est_cat.to_full_params()
    if cfg.predict.plot.show_plot and (plocs is not None):
        ptc = cfg.encoder.tiles_to_crop * cfg.encoder.tile_slen
        cropped_image = images[0, 0, ptc:-ptc, ptc:-ptc]
        cropped_background = background[0, 0, ptc:-ptc, ptc:-ptc]
        plot_predict(
            cfg,
            cropped_image,
            cropped_background,
            plocs,
            est_full,
            survey_obj["wcs"][SDSS.BANDS.index("r")],
        )

    return est_full, images[0], background[0], plocs, pred


def crop_plocs(cfg, w, h, plocs, do_crop=False):
    tiles_to_crop = cfg.encoder.tiles_to_crop
    tile_slen = cfg.encoder.tile_slen
    minh = tiles_to_crop * tile_slen
    maxh = h + tiles_to_crop * tile_slen
    minw = tiles_to_crop * tile_slen
    maxw = w + tiles_to_crop * tile_slen
    if do_crop:
        minh += cfg.predict.crop.left_upper_corner[0]
        maxh += cfg.predict.crop.left_upper_corner[0]
        minw += cfg.predict.crop.left_upper_corner[1]
        maxw += cfg.predict.crop.left_upper_corner[1]

    x0_mask = (plocs[:, 0] > minh) & (plocs[:, 0] < maxh)
    x1_mask = (plocs[:, 1] > minw) & (plocs[:, 1] < maxw)
    x_mask = x0_mask * x1_mask

    return plocs[x_mask].cpu() - torch.tensor([minh, minw])


def add_cat(p, est_plocs, true_plocs, decals_plocs):
    """Function to overlay scatter plots on given image using different catalogs."""
    p.scatter(
        est_plocs[:, 1],
        est_plocs[:, 0],
        color="darksalmon",
        marker="circle",
        legend_label="estimated catalog",
        size=10,
        fill_color=None,
    )
    p.scatter(
        true_plocs[:, 1],
        true_plocs[:, 0],
        marker="circle",
        color="cyan",
        legend_label="sdss catalog",
        size=20,
        fill_color=None,
    )
    if decals_plocs is not None:
        p.scatter(
            decals_plocs[:, 1],
            decals_plocs[:, 0],
            marker="circle",
            color="lightgreen",
            legend_label="decals catalog",
            size=5,
        )
    p.legend.click_policy = "hide"
    return p


def plot_image(cfg, img, w, h, est_plocs, true_plocs, title):
    """Function that generate plots for images."""
    p = figure(width=cfg.predict.plot.width, height=cfg.predict.plot.height)
    p.image(image=[img], x=0, y=0, dw=w, dh=h, palette="Viridis256")
    do_crop = cfg.predict.crop.do_crop
    is_simulated = cfg.predict.is_simulated

    true_plocs = np.array(crop_plocs(cfg, w, h, true_plocs, do_crop).cpu())
    decals_plocs = None
    if do_crop and (not is_simulated):
        sdss = instantiate(cfg.predict.dataset)
        wcs = sdss[0]["wcs"][0]

        # Map SDSS RCF to DECaLS brickname
        sdss_fields = cfg.surveys.sdss.sdss_fields
        run, camcol, fields = sdss_fields[0].values()
        brickname = DECaLS.brick_for_radec(*SDSS.radec_for_rcf(run, camcol, fields[0]))

        tractor_filename = DecalsDownloader(brickname, cfg.paths.decals).download_catalog()
        decals_plocs_from_sdss = DecalsFullCatalog.from_file(
            tractor_filename, wcs, height=sdss[0]["image"].shape[1], width=sdss[0]["image"].shape[2]
        ).plocs[0]
        decals_plocs = np.array(crop_plocs(cfg, w, h, decals_plocs_from_sdss, do_crop))
    return TabPanel(child=add_cat(p, est_plocs, true_plocs, decals_plocs), title=title)


def plot_predict(cfg, image, background, true_plocs, est_cat: FullCatalog, wcs: WCS):
    """Function that uses bokeh to save generated plots to an html file."""
    w, h = image.shape

    est_plocs = np.array(est_cat.plocs.cpu())[0]
    simulator = instantiate(cfg.simulator)
    decoder_obj = simulator.image_decoder
    est_tile = est_cat.to_tile_params(
        cfg.encoder.tile_slen, cfg.simulator.survey.prior_config.max_sources
    )
    # convert plocs to RA, Dec using wcs
    # ras, decs = wcs.all_pix2world(est_plocs[:, 1], est_plocs[:, 0], 0) # noqa: E800
    # ra_lim, dec_lim = (np.min(ras), np.max(ras)), (np.min(decs), np.max(decs)) # noqa: E800
    # rcfs = np.array([SDSS.rcf_for_radec(ra_lim, dec_lim)]) # noqa: E800
    # TODO: don't hardcode 94, 1, 12. Approach above yields different RCF...
    rcfs = np.array([[94, 1, 12]])
    images, _, _ = decoder_obj.render_images(est_tile.to("cpu"), rcfs)
    recon_img = images[0][0]  # first image in batch, first band in image

    image = image.to("cpu")
    background = background.to("cpu")
    res = image - recon_img - background

    # normalize for big image
    title = ""
    if w >= 150:
        image = np.log((image - image.min()) + 10)
        recon_img = np.log((recon_img - recon_img.min()) + 10)
        title = "log-"

    if cfg.predict.plot.out_file_name is not None:
        # Create parent folder
        Path(cfg.predict.plot.out_file_name).parent.mkdir(parents=True, exist_ok=True)
        output_file(cfg.predict.plot.out_file_name)

    np_image = np.array(image)
    np_recon = np.array(recon_img)
    tab1 = plot_image(cfg, np_image, w, h, est_plocs, true_plocs, title + "true image")
    tab2 = plot_image(cfg, np_recon, w, h, est_plocs, true_plocs, title + "reconstruct image")
    tab3 = plot_image(cfg, np.array(res), w, h, est_plocs, true_plocs, "residual")
    show(Tabs(tabs=[tab1, tab2, tab3]))
