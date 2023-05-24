import numpy as np
import torch
from bokeh.models import TabPanel, Tabs
from bokeh.plotting import figure, output_file, show
from hydra.utils import instantiate

from bliss.surveys.decals import DecalsFullCatalog
from bliss.surveys.sdss import PhotoFullCatalog


def prepare_image(x, device):
    x = torch.from_numpy(x).unsqueeze(0)
    x = x.to(device=device)
    # image dimensions must be a multiple of 16
    height = x.size(2) - (x.size(2) % 16)
    width = x.size(3) - (x.size(3) % 16)
    return x[:, :, :height, :width]


def crop_plocs(cfg, h, plocs):
    tiles_to_crop = cfg.encoder.tiles_to_crop
    tile_slen = cfg.encoder.tile_slen
    if not cfg.predict.ifSimulated:
        crop = cfg.predict.crop
        minb = crop[0] + tiles_to_crop * tile_slen
        maxb = crop[1] - tiles_to_crop * tile_slen

    else:
        minb = tiles_to_crop * tile_slen
        maxb = h + tiles_to_crop * tile_slen

    x0_mask = (plocs[:, 0] > minb) & (plocs[:, 0] < maxb)
    x1_mask = (plocs[:, 1] > minb) & (plocs[:, 1] < maxb)
    x_mask = x0_mask * x1_mask

    return plocs[x_mask] - minb


def add_cat(p, est_plocs, true_plocs, decals_plocs=None):
    p.scatter(
        est_plocs[:, 1],
        est_plocs[:, 0],
        color="red",
        marker="circle",
        legend_label="est cat",
        size=10,
        fill_color=None,
    )
    p.scatter(
        true_plocs[:, 1],
        true_plocs[:, 0],
        marker="circle",
        color="green",
        legend_label="sdss cat",
        size=20,
        fill_color=None,
    )
    if decals_plocs is not None:
        p.scatter(
            decals_plocs[:, 1],
            decals_plocs[:, 0],
            marker="circle",
            color="pink",
            legend_label="decals cat",
            size=5,
        )
    p.legend.click_policy = "hide"
    return p


def plot_image(cfg, img, w, h, est_plocs, true_plocs, title):
    p = figure(width=cfg.predict.plot.width, height=cfg.predict.plot.height)
    p.image(image=[img], x=0, y=0, dw=w, dh=h, palette="Viridis256")
    if cfg.predict.ifSimulated:
        tab = TabPanel(child=add_cat(p, est_plocs, true_plocs), title=title)
    else:
        decals_plocs = np.array(crop_plocs(cfg, h, decal_plocs_from_sdss(cfg)[0]))
        tab = TabPanel(child=add_cat(p, est_plocs, true_plocs, decals_plocs), title=title)
    return tab


def plot_predict(cfg, image, background, true_plocs, est_cat):
    w, h = image.shape

    est_plocs = np.array(est_cat.plocs.cpu())[0]
    true_plocs = np.array(crop_plocs(cfg, h, true_plocs).cpu())
    decoder_obj = instantiate(cfg.simulator.decoder)
    est_tile = est_cat.to_tile_params(cfg.encoder.tile_slen, cfg.simulator.prior.max_sources)
    recon_img = decoder_obj.render_images(est_tile.cpu())[0, 0]
    res = torch.tensor(image).cpu() - recon_img - torch.tensor(background).cpu()

    # normalize for big image
    title = ""
    if w >= 200:
        image = np.log((image - image.min()).cpu() + 10)
        recon_img = np.log((recon_img - recon_img.min()).cpu() + 10)
        title = "log-"

    output_file(cfg.predict.plot.out_file_name)

    np_image = np.array(image.cpu())
    np_recon = np.array(recon_img)
    tab1 = plot_image(cfg, np_image, w, h, est_plocs, true_plocs, title + "image_true")
    tab2 = plot_image(cfg, np_recon, w, h, est_plocs, true_plocs, title + "recon")
    tab3 = plot_image(cfg, np.array(res), w, h, est_plocs, true_plocs, title + "res")
    show(Tabs(tabs=[tab1, tab2, tab3]))


def predict(cfg, image, background, true_plocs):
    encoder = instantiate(cfg.encoder).to(cfg.predict.device)
    enc_state_dict = torch.load(cfg.predict.weight_save_path)
    encoder.load_state_dict(enc_state_dict)
    encoder.eval()

    batch = {"images": image, "background": background}

    with torch.no_grad():
        pred = encoder.encode_batch(batch)
        est_cat = encoder.variational_mode(pred)

    if cfg.predict.plot.ifPlot:
        ttc = cfg.encoder.tiles_to_crop
        ts = cfg.encoder.tile_slen
        ptc = ttc * ts
        cropped_image = image[0, 0, ptc:-ptc, ptc:-ptc]
        cropped_background = background[0, 0, ptc:-ptc, ptc:-ptc]
        plot_predict(cfg, cropped_image, cropped_background, true_plocs, est_cat)

    return est_cat


def decal_plocs_from_sdss(cfg):
    sdss = instantiate(cfg.predict.dataset)
    idx0, idx1 = cfg.predict.crop[0], cfg.predict.crop[1]
    wcs = sdss[0]["wcs"][0]
    ra_lim, dec_lim = wcs.all_pix2world((idx0, idx1), (idx0, idx1), 0)

    decals_path = cfg.paths.decals + "/tractor-3366m010.fits"
    cat = DecalsFullCatalog.from_file(decals_path, ra_lim, dec_lim)

    return cat.get_plocs_from_ra_dec(wcs)


def predict_sdss(cfg):
    idx0, idx1 = cfg.predict.crop[0], cfg.predict.crop[1]
    sdss_plocs = PhotoFullCatalog.from_file(
        cfg.paths.sdss,
        run=cfg.predict.dataset.run,
        camcol=cfg.predict.dataset.camcol,
        field=cfg.predict.dataset.fields[0],
        band=cfg.predict.dataset.bands[0],
    ).plocs[0]
    sdss = instantiate(cfg.predict.dataset)
    crop_img = sdss[0]["image"][:, idx0:idx1, idx0:idx1]
    crop_bg = sdss[0]["background"][:, idx0:idx1, idx0:idx1]

    prepare_img = prepare_image(crop_img, cfg.predict.device)
    prepare_bg = prepare_image(crop_bg, cfg.predict.device)

    return predict(cfg, prepare_img, prepare_bg, sdss_plocs), crop_img, crop_bg
