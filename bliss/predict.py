import numpy as np
import torch
from bokeh.models import TabPanel, Tabs
from bokeh.plotting import figure, output_file, show
from hydra.utils import instantiate

from bliss.surveys.decals import DecalsFullCatalog
from bliss.surveys.sdss import PhotoFullCatalog


def predict_sdss(cfg):
    sdss_plocs = PhotoFullCatalog.from_file(
        cfg.paths.sdss,
        run=cfg.predict.dataset.run,
        camcol=cfg.predict.dataset.camcol,
        field=cfg.predict.dataset.fields[0],
        band=cfg.predict.dataset.bands[0],
    ).plocs[0]
    sdss = instantiate(cfg.predict.dataset)

    encoder = instantiate(cfg.encoder).to(cfg.predict.device)
    enc_state_dict = torch.load(cfg.predict.weight_save_path)
    encoder.load_state_dict(enc_state_dict)
    encoder.eval()
    trainer = instantiate(cfg.predict.trainer)
    est_cat, images, background, pred = trainer.predict(encoder, datamodule=sdss)[0].values()
    if cfg.predict.plot.show_plot and (sdss_plocs is not None):
        ptc = cfg.encoder.tiles_to_crop * cfg.encoder.tile_slen
        cropped_image = images[0, 0, ptc:-ptc, ptc:-ptc]
        cropped_background = background[0, 0, ptc:-ptc, ptc:-ptc]
        plot_predict(cfg, cropped_image, cropped_background, sdss_plocs, est_cat)

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
        output_file(cfg.predict.plot.out_file_name)

    np_image = np.array(image)
    np_recon = np.array(recon_img)
    tab1 = plot_image(cfg, np_image, w, h, est_plocs, true_plocs, title + "true image")
    tab2 = plot_image(cfg, np_recon, w, h, est_plocs, true_plocs, title + "reconstruct image")
    tab3 = plot_image(cfg, np.array(res), w, h, est_plocs, true_plocs, "residual")
    show(Tabs(tabs=[tab1, tab2, tab3]))
