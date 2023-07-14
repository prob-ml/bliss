from pathlib import Path

import numpy as np
import torch
from bokeh.models import TabPanel, Tabs
from bokeh.plotting import figure, output_file, show
from hydra.utils import instantiate
from reproject import reproject_interp
from torch import Tensor

from bliss.catalog import FullCatalog
from bliss.surveys.decals import DarkEnergyCameraLegacySurvey as DECaLS
from bliss.surveys.decals import DecalsFullCatalog
from bliss.surveys.sdss import PhotoFullCatalog
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

    top_left_y = crop_params.left_upper_corner[0]
    top_left_x = crop_params.left_upper_corner[1]
    width = crop_params.width
    height = crop_params.height
    if ((top_left_y + height) <= image.shape[2]) and ((top_left_x + width) <= image.shape[3]):
        image = image[:, :, top_left_y : top_left_y + height, top_left_x : top_left_x + width]
        background = background[
            :, :, top_left_y : top_left_y + height, top_left_x : top_left_x + width
        ]
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


def predict(cfg):
    survey = instantiate(cfg.predict.dataset)

    # below collections indexed by image_id
    images_for_frame = {}
    radecs_for_frame = {}
    backgrounds_for_frame = {}
    preds_for_frame = {}

    plocs_all = None
    est_full_all = None  # collated catalog for all images
    survey_objs = [survey[i] for i in range(len(survey))]
    for i, survey_obj in enumerate(survey_objs):
        survey_obj["image"] = align(
            survey_obj["image"],
            ref_wcs=survey_obj["wcs"],
            ref_band=cfg.predict.dataset.reference_band,
        )
        survey_obj["background"] = align(
            survey_obj["background"],
            ref_wcs=survey_obj["wcs"],
            ref_band=cfg.predict.dataset.reference_band,
        )

        cat_path = survey.downloader.download_catalog(survey.image_id(i))
        plocs = survey.catalog_cls.from_file(
            cat_path=cat_path,
            wcs=survey_obj["wcs"][cfg.predict.dataset.reference_band],
            height=survey_obj["image"].shape[1],
            width=survey_obj["image"].shape[2],
        ).plocs[0]

        # get RA, Dec of the center of the image
        ra, dec = survey_obj["wcs"][cfg.predict.dataset.reference_band].all_pix2world(
            survey_obj["image"].shape[2] / 2, survey_obj["image"].shape[1] / 2, 0
        )
        radecs_for_frame[survey.image_id(i)] = (ra.item(), dec.item())

        encoder = instantiate(cfg.encoder).to(cfg.predict.device)
        enc_state_dict = torch.load(cfg.predict.weight_save_path)
        encoder.load_state_dict(enc_state_dict)
        encoder.eval()
        trainer = instantiate(cfg.predict.trainer)
        images = prepare_image(survey_obj["image"], cfg.predict.device).float()
        backgrounds = prepare_image(survey_obj["background"], cfg.predict.device).float()
        images, backgrounds = crop_image(images, backgrounds, cfg.predict.crop)
        survey.predict_batch = prepare_batch(images, backgrounds)
        est_cat, images, backgrounds, pred = trainer.predict(encoder, datamodule=survey)[0].values()

        # mean of the nelec_per_mgy per band
        nelec_per_nmgy_per_band = np.mean(survey_obj["nelec_per_nmgy_list"], axis=1)
        est_cat = nelec_to_nmgy_for_catalog(est_cat, nelec_per_nmgy_per_band)
        est_full = est_cat.to_full_params()

        images_for_frame[survey.image_id(i)] = images
        backgrounds_for_frame[survey.image_id(i)] = backgrounds
        preds_for_frame[survey.image_id(i)] = pred

        if plocs_all is None:
            plocs_all = plocs
        else:
            plocs_all = torch.cat((plocs_all, plocs), dim=0)
            plocs_all = torch.unique(plocs_all, dim=0)

        if not est_full_all:
            est_full_all = est_full
        else:
            d = {}
            d["plocs"] = torch.cat((est_full_all.plocs, est_full.plocs), dim=1)
            d["n_sources"] = Tensor([est_full_all.n_sources + est_full.n_sources])
            est_full_all_dict = est_full_all.to_dict()
            for k, v in est_full.items():
                d[k] = torch.cat((est_full_all_dict[k], v), dim=1)
            est_full_all = FullCatalog(est_full_all.height, est_full_all.width, d)

    assert est_full_all is not None and isinstance(
        est_full_all, FullCatalog
    ), "Should have estimated catalog for at least one image"
    if cfg.predict.plot.show_plot and (plocs_all is not None):
        plot_predict(
            cfg,
            images_for_frame,
            backgrounds_for_frame,
            radecs_for_frame,
            plocs_all,
            est_full_all,
        )

    images_for_frame = {k: v[0] for k, v in images_for_frame.items()}
    backgrounds_for_frame = {k: v[0] for k, v in backgrounds_for_frame.items()}
    return est_full_all, images_for_frame, backgrounds_for_frame, plocs_all, preds_for_frame


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


def add_cat(p, est_plocs, survey_true_plocs, sdss_plocs, decals_plocs):
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
        survey_true_plocs[:, 1],
        survey_true_plocs[:, 0],
        marker="circle",
        color="hotpink",
        legend_label="consolidated survey catalog",
        size=20,
        fill_color=None,
    )
    if sdss_plocs is not None:
        p.scatter(
            sdss_plocs[:, 1],
            sdss_plocs[:, 0],
            marker="circle",
            color="lightgreen",
            legend_label="sdss catalog",
            size=5,
            fill_color=None,
        )
    if decals_plocs is not None:
        p.scatter(
            decals_plocs[:, 1],
            decals_plocs[:, 0],
            marker="circle",
            color="cyan",
            legend_label="decals catalog",
            size=5,
        )
    p.legend.click_policy = "hide"
    return p


def plot_image(cfg, ra, dec, img, w, h, est_plocs, survey_true_plocs, title):
    """Function that generate plots for images."""
    p = figure(width=cfg.predict.plot.width, height=cfg.predict.plot.height)
    p.image(image=[img], x=0, y=0, dw=w, dh=h, palette="Viridis256")
    do_crop = cfg.predict.crop.do_crop
    is_simulated = cfg.predict.is_simulated

    survey_true_plocs = np.array(crop_plocs(cfg, w, h, survey_true_plocs, do_crop).cpu())
    sdss_plocs = None
    decals_plocs = None
    if do_crop and (not is_simulated):
        # DECaLS one-instance catalog plocs
        decals = instantiate(
            cfg.surveys.decals, bands=[SDSS.BANDS.index("r")], sky_coords=[{"ra": ra, "dec": dec}]
        )

        brickname = DECaLS.brick_for_radec(ra, dec)
        tractor_filename = decals.downloader.download_catalog(brickname)
        decals_plocs = DecalsFullCatalog.from_file(
            tractor_filename,
            wcs=decals[0]["wcs"][cfg.predict.dataset.reference_band],
            height=decals[0]["image"].shape[1],
            width=decals[0]["image"].shape[2],
        ).plocs[0]
        decals_plocs = np.array(crop_plocs(cfg, w, h, decals_plocs, do_crop).cpu())

        # SDSS one-instance catalog plocs
        run, camcol, field = SDSS.rcf_for_radec(ra, dec)
        sdss = instantiate(
            cfg.surveys.sdss, sdss_fields=[{"run": run, "camcol": camcol, "fields": [field]}]
        )
        photocat_filename = sdss.downloader.download_catalog((run, camcol, field))
        sdss_plocs = PhotoFullCatalog.from_file(
            photocat_filename,
            wcs=sdss[0]["wcs"][cfg.predict.dataset.reference_band],
            height=sdss[0]["image"].shape[1],
            width=sdss[0]["image"].shape[2],
        ).plocs[0]
        sdss_plocs = np.array(crop_plocs(cfg, w, h, sdss_plocs, do_crop).cpu())
    return TabPanel(
        child=add_cat(p, est_plocs, survey_true_plocs, sdss_plocs, decals_plocs), title=title
    )


def plot_predict(
    cfg,
    images_for_frame,
    backgrounds_for_frame,
    radecs_for_frame,
    survey_true_plocs,
    est_cat: FullCatalog,
):
    """Function that uses bokeh to save generated plots to an html file."""
    if cfg.predict.plot.out_file_name is not None:
        out_filepath = Path(cfg.predict.plot.out_file_name)
        out_filepath.parent.mkdir(parents=True, exist_ok=True)
        output_file(out_filepath)

    est_plocs = np.array(est_cat.plocs.cpu())[0]
    est_tile = est_cat.to_tile_params(
        cfg.encoder.tile_slen,
        cfg.simulator.survey.prior_config.max_sources,
        ignore_extra_sources=True,
    ).to("cpu")

    tabs = []
    for image_id in images_for_frame.keys():
        image = images_for_frame[image_id]
        background = backgrounds_for_frame[image_id]

        ptc = cfg.encoder.tiles_to_crop * cfg.encoder.tile_slen
        image = image[0, 0, ptc:-ptc, ptc:-ptc]
        background = background[0, 0, ptc:-ptc, ptc:-ptc]

        w, h = image.shape

        ra, dec = radecs_for_frame[image_id]
        run, camcol, field = SDSS.rcf_for_radec(ra, dec)
        simulator = instantiate(
            cfg.simulator,
            survey={"sdss_fields": [{"run": run, "camcol": camcol, "fields": [field]}]},
        )
        decoder_obj = simulator.image_decoder
        recon_images, _, _ = decoder_obj.render_images(est_tile, np.array([[run, camcol, field]]))
        recon_img = recon_images[0][0]  # first image in batch, first band in image

        image = image.to("cpu")
        background = background.to("cpu")
        res = image - recon_img - background

        # normalize for big image
        title_suffix = f" (RA: {float(ra):.2f}, Dec: {float(dec):.2f})"
        title_prefix = ""
        if w >= 150:
            image = np.log((image - image.min()) + 10)
            recon_img = np.log((recon_img - recon_img.min()) + 10)
            title_prefix = "log-"

        np_image = np.array(image)
        np_recon = np.array(recon_img)
        np_res = np.array(res)
        tab1_title = f"{title_prefix}true image{title_suffix}"
        tab2_title = f"{title_prefix}reconstructed image{title_suffix}"
        tab3_title = f"residual{title_suffix}"
        tab1 = plot_image(cfg, ra, dec, np_image, w, h, est_plocs, survey_true_plocs, tab1_title)
        tab2 = plot_image(cfg, ra, dec, np_recon, w, h, est_plocs, survey_true_plocs, tab2_title)
        tab3 = plot_image(cfg, ra, dec, np_res, w, h, est_plocs, survey_true_plocs, tab3_title)
        tabs.extend([tab1, tab2, tab3])

    show(Tabs(tabs=tabs))
