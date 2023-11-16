from pathlib import Path

import numpy as np
import torch
from bokeh.models import TabPanel, Tabs
from bokeh.plotting import figure, output_file, show
from hydra.utils import instantiate

from bliss.catalog import FullCatalog
from bliss.surveys.sdss import PhotoFullCatalog
from bliss.surveys.sdss import SloanDigitalSkySurvey as SDSS


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
    if survey_true_plocs is not None:
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
    from bliss.surveys.decals import DECaLS  # pylint: disable=import-outside-toplevel
    from bliss.surveys.des import TractorFullCatalog  # pylint: disable=import-outside-toplevel

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
            cfg.surveys.decals, bands=[DECaLS.BANDS.index("r")], sky_coords=[{"ra": ra, "dec": dec}]
        )
        brickname = DECaLS.brick_for_radec(ra, dec)
        tractor_filename = decals.downloader.download_catalog(brickname)
        decals_plocs = TractorFullCatalog.from_file(
            tractor_filename,
            wcs=decals[0]["wcs"][cfg.simulator.prior.reference_band],
            height=decals[0]["background"].shape[1],
            width=decals[0]["background"].shape[2],
        ).plocs[0]
        decals_plocs = np.array(crop_plocs(cfg, w, h, decals_plocs, do_crop).cpu())

        # SDSS one-instance catalog plocs
        run, camcol, field = SDSS.rcf_for_radec(ra, dec)
        sdss = instantiate(
            cfg.surveys.sdss, fields=[{"run": run, "camcol": camcol, "fields": [field]}]
        )
        photocat_filename = sdss.downloader.download_catalog((run, camcol, field))
        sdss_plocs = PhotoFullCatalog.from_file(
            photocat_filename,
            wcs=sdss[0]["wcs"][cfg.simulator.prior.reference_band],
            height=sdss[0]["background"].shape[1],
            width=sdss[0]["background"].shape[2],
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
    est_tile = est_cat.to_tile_catalog(
        cfg.encoder.tile_slen,
        cfg.simulator.prior.max_sources,
        ignore_extra_sources=True,
    ).to("cpu")

    tabs = []
    for image_id in images_for_frame.keys():
        image = images_for_frame[image_id]
        background = backgrounds_for_frame[image_id]

        ptc = cfg.encoder.tiles_to_crop * cfg.encoder.tile_slen
        image = image[0, ptc:-ptc, ptc:-ptc]  # uh, are we always plotting the u band image here?
        background = background[0, ptc:-ptc, ptc:-ptc]

        w, h = image.shape

        ra, dec = radecs_for_frame[image_id]
        run, camcol, field = SDSS.rcf_for_radec(ra, dec)
        simulator = instantiate(
            cfg.simulator,
            survey={"fields": [{"run": run, "camcol": camcol, "fields": [field]}]},
        )
        decoder_obj = simulator.image_decoder
        recon_images = decoder_obj.render_images(est_tile, [(run, camcol, field)])[
            0
        ]  # NOTE: causes issue when weights used were for survey that has diff no. of bands as SDSS
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

        # we are plotting detections from all frames on each frame here, but we shouldn't be!
        # each frame has it's own bliss catalog and it's own survey catalog
        tab1 = plot_image(cfg, ra, dec, np_image, w, h, est_plocs, survey_true_plocs, tab1_title)
        tab2 = plot_image(cfg, ra, dec, np_recon, w, h, est_plocs, survey_true_plocs, tab2_title)
        tab3 = plot_image(cfg, ra, dec, np_res, w, h, est_plocs, survey_true_plocs, tab3_title)
        tabs.extend([tab1, tab2, tab3])

    show(Tabs(tabs=tabs))
