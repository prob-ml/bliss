from pathlib import Path

import numpy as np
import torch
from bokeh.models import TabPanel, Tabs
from bokeh.plotting import figure, output_file, show
from hydra.utils import instantiate
from reproject import reproject_interp

from bliss.surveys.decals import DecalsFullCatalog
from bliss.surveys.sdss import PhotoFullCatalog


def prepare_image(x, device):
    x = torch.from_numpy(x).unsqueeze(0)
    x = x.to(device=device)
    # image dimensions must be a multiple of 16
    height = x.size(2) - (x.size(2) % 16)
    width = x.size(3) - (x.size(3) % 16)
    x = x[:, :, :height, :width]

    return x  # noqa: WPS331


def align(img, sdss):
    """Reproject images based on some reference WCS for pixel alignment."""
    reproj_d = {}
    footprint_d = {}

    orig_dim = img.shape

    # align with r-band WCS
    for bnd in range(img.shape[0]):
        inputs = (img[bnd], sdss[0]["wcs"][bnd])
        reproj, footprint = reproject_interp(  # noqa: WPS317
            inputs, sdss[0]["wcs"][2], order="bicubic", shape_out=img[bnd].shape
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


def predict_sdss(cfg):
    sdss = instantiate(cfg.predict.dataset)

    sdss_frame = PhotoFullCatalog.from_file(
        cfg.paths.sdss,
        run=cfg.predict.dataset.run,
        camcol=cfg.predict.dataset.camcol,
        field=cfg.predict.dataset.fields[0],
        sdss_obj=sdss,
    )

    sdss_plocs = sdss_frame.plocs[0]
    img = sdss[0]["image"]
    bg = sdss[0]["background"]
    img = align(img, sdss)
    bg = align(bg, sdss)

    encoder = instantiate(cfg.encoder).to(cfg.predict.device)
    enc_state_dict = torch.load(cfg.predict.weight_save_path)
    encoder.load_state_dict(enc_state_dict)
    encoder.eval()
    trainer = instantiate(cfg.predict.trainer)
    est_cat, images, background, pred = trainer.predict(encoder, datamodule=sdss)[0].values()
    est_full = est_cat.to_full_params()
    if cfg.predict.plot.show_plot and (sdss_plocs is not None):
        ptc = cfg.encoder.tiles_to_crop * cfg.encoder.tile_slen
        cropped_image = images[0, 0, ptc:-ptc, ptc:-ptc]
        cropped_background = background[0, 0, ptc:-ptc, ptc:-ptc]
        plot_predict(cfg, cropped_image, cropped_background, sdss_plocs, est_full)

    return est_cat, images[0], background[0], sdss_plocs, pred


def decal_plocs_from_sdss(cfg):
    """Use wcs to match sdss image with corresponding DeCals catalog for futher plotting."""
    sdss = instantiate(cfg.predict.dataset)
    idx0 = cfg.predict.crop.left_upper_corner[0]
    idx1 = cfg.predict.crop.left_upper_corner[1]
    width = cfg.predict.crop.width
    height = cfg.predict.crop.height
    wcs = sdss[0]["wcs"][0]
    ra_lim, dec_lim = wcs.all_pix2world((idx1, idx1 + width), (idx0, idx0 + height), 0)

    decals_path = cfg.paths.decals + "/tractor-3366m010.fits"
    cat = DecalsFullCatalog.from_file(decals_path, ra_lim, dec_lim)

    return cat.get_plocs_from_ra_dec(wcs)


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
    """Function that overlaying scatter plots on given image using different catalogs."""
    p.scatter(
        est_plocs[:, 1],
        est_plocs[:, 0],
        color="darksalmon",
        marker="circle",
        legend_label="estimate catalog",
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
        decals_plocs = np.array(crop_plocs(cfg, w, h, decal_plocs_from_sdss(cfg)[0], do_crop))
    return TabPanel(child=add_cat(p, est_plocs, true_plocs, decals_plocs), title=title)


def plot_predict(cfg, image, background, true_plocs, est_cat):
    """Function that uses bokeh to save generated plots to an html file."""
    w, h = image.shape

    est_plocs = np.array(est_cat.plocs.cpu())[0]
    decoder_obj = instantiate(cfg.simulator.decoder)
    est_tile = est_cat.to_tile_params(cfg.encoder.tile_slen, cfg.simulator.prior.max_sources)
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
