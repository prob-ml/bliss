import os
from typing import Tuple
from einops.einops import rearrange
import torch
import numpy as np
from matplotlib import pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from astropy.table import Table

from bliss.datasets import sdss
from bliss import reporting
from bliss.models.binary import BinaryEncoder
from bliss.models.galaxy_encoder import GalaxyEncoder
from bliss.sleep import SleepPhase
from bliss.predict import predict_on_image


# class SDSSReconstructionFigures:


def compute_data(
    sdss_data: dict,
    coadd_cat,
    lims: Tuple[int, int],
    sleep_net: SleepPhase,
    binary_encoder: BinaryEncoder,
    galaxy_encoder: GalaxyEncoder,
):
    scene = sdss_data["image"]
    # coadd_params = reporting.get_params_from_coadd(coadd_cat, h, w, bp)
    assert isinstance(scene, (torch.Tensor, np.ndarray))
    assert sleep_net.device == binary_encoder.device == galaxy_encoder.device
    device = sleep_net.device

    bp = 24

    image_encoder = sleep_net.image_encoder.to(device).eval()
    image_decoder = sleep_net.image_decoder.to(device).eval()

    bp = image_encoder.border_padding
    xlim, ylim = lims
    h, w = ylim[1] - ylim[0], xlim[1] - xlim[0]
    assert h >= bp and w >= bp
    hb = h + 2 * bp
    wb = w + 2 * bp
    chunk = scene[ylim[0] - bp : ylim[1] + bp, xlim[0] - bp : xlim[1] + bp]
    chunk = torch.from_numpy(chunk.reshape(1, 1, hb, wb)).to(device)

    # for plotting
    chunk_np = chunk.reshape(hb, wb).cpu().numpy()

    with torch.no_grad():

        tile_map, full_map, _ = predict_on_image(
            chunk, image_encoder, binary_encoder, galaxy_encoder
        )

        # plot image from tile est.
        recon_image, _ = image_decoder.render_images(
            tile_map["n_sources"],
            tile_map["locs"],
            tile_map["galaxy_bool"],
            tile_map["galaxy_params"],
            tile_map["fluxes"],
            add_noise=False,
        )

    recon_image = recon_image.cpu().numpy().reshape(hb, wb)

    # only keep section inside obrder padding
    true_image = chunk_np[bp : hb - bp, bp : wb - bp]
    recon_image = recon_image[bp : hb - bp, bp : wb - bp]
    residual = (true_image - recon_image) / np.sqrt(recon_image)

    locs_true = torch.stack(
        (
            torch.from_numpy(np.array(coadd_cat["x"]).astype(float)),
            torch.from_numpy(np.array(coadd_cat["y"]).astype(float)),
        ),
        dim=1,
    )
    locs_true = locs_true[xlim[0] < locs_true[:, 0]]
    locs_true = locs_true[ylim[0] < locs_true[:, 1]]

    locs_true = locs_true[locs_true[:, 0] < xlim[1]]
    locs_true = locs_true[locs_true[:, 1] < ylim[1]]

    ## Shift locs by lower limit
    locs_true[:, 0] = locs_true[:, 0] - xlim[0]
    locs_true[:, 1] = locs_true[:, 1] - ylim[0]

    locs_pred = rearrange(full_map["plocs"], "a b c -> (a b) c").cpu().numpy()

    data = (true_image, recon_image, residual, locs_true, locs_pred)

    return data


def create_figure(data):
    """Make figures related to detection and classification in SDSS."""
    plt.style.use("seaborn-colorblind")
    pad = 6.0
    reporting.set_rc_params(fontsize=22, tick_label_size="small", legend_fontsize="small")
    # for figname in self.fignames:
    true, recon, res, locs, locs_pred = data
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(28, 12))
    assert len(true.shape) == len(recon.shape) == len(res.shape) == 2

    # pick standard ranges for residuals
    vmin_res, vmax_res = res.min().item(), res.max().item()

    ax_true = axes[0]
    ax_recon = axes[1]
    ax_res = axes[2]

    ax_true.set_title("Original Image", pad=pad)
    ax_recon.set_title("Reconstruction", pad=pad)
    ax_res.set_title("Residual", pad=pad)

    # plot images
    reporting.plot_image(fig, ax_true, true, vrange=(800, 1200))
    reporting.plot_image(fig, ax_recon, recon, vrange=(800, 1200))
    reporting.plot_image(fig, ax_res, res, vrange=(vmin_res, vmax_res))

    # reporting.plot_locs(ax_true, slen=53, bpad=24, locs=locs)
    ax_true.scatter(locs[:, 0], locs[:, 1], color="r", marker="x", s=20)
    ax_recon.scatter(locs[:, 0], locs[:, 1], color="r", marker="x", s=20)
    ax_res.scatter(locs[:, 0], locs[:, 1], color="r", marker="x", s=20)

    ax_true.scatter(locs_pred[:, 1], locs_pred[:, 0], color="b", marker="x", s=20)
    ax_recon.scatter(locs_pred[:, 1], locs_pred[:, 0], color="b", marker="x", s=20)
    ax_res.scatter(locs_pred[:, 1], locs_pred[:, 0], color="b", marker="x", s=20)

    plt.subplots_adjust(hspace=-0.4)
    plt.tight_layout()

    return fig


files_dict = {
    "sleep_ckpt": "models/sdss_sleep.ckpt",
    "galaxy_encoder_ckpt": "models/sdss_galaxy_encoder.ckpt",
    "binary_ckpt": "models/sdss_binary.ckpt",
    "ae_ckpt": "models/sdss_autoencoder.ckpt",
    "coadd_cat": "data/coadd_catalog_94_1_12.fits",
    "sdss_dir": "data/sdss",
    "psf_file": "data/psField-000094-1-0012-PSF-image.npy",
}
device = torch.device("cuda:0")


def get_sdss_data(sdss_pixel_scale=0.396):
    run = 94
    camcol = 1
    field = 12
    bands = (2,)
    sdss_data = sdss.SloanDigitalSkySurvey(
        sdss_dir=files_dict["sdss_dir"],
        run=run,
        camcol=camcol,
        fields=(field,),
        bands=bands,
        overwrite_cache=True,
        overwrite_fits_cache=True,
    )

    return {
        "image": sdss_data[0]["image"][0],
        "wcs": sdss_data[0]["wcs"][0],
        "pixel_scale": sdss_pixel_scale,
    }


lims = {
    "sdss_recon0": ((1700, 2000), (200, 500)),  # scene
    "sdss_recon1": ((1000, 1300), (1150, 1450)),  # scene
    "sdss_recon2": ((742, 790), (460, 508)),  # individual blend
    "sdss_recon3": ((1128, 1160), (25, 57)),  # individual blend
    "sdss_recon4": ((500, 552), (170, 202)),  # individual blend
}


def main_plotly(model_name="sdss_recon0"):
    os.chdir(os.getenv("BLISS_HOME"))  # simplicity for I/O

    sleep_net = SleepPhase.load_from_checkpoint(files_dict["sleep_ckpt"]).to(device)
    binary_encoder = BinaryEncoder.load_from_checkpoint(files_dict["binary_ckpt"])
    binary_encoder = binary_encoder.to(device).eval()
    galaxy_encoder = GalaxyEncoder.load_from_checkpoint(files_dict["galaxy_encoder_ckpt"])
    galaxy_encoder = galaxy_encoder.to(device).eval()

    # FIGURE 3: Reconstructions on SDSS
    sdss_data = get_sdss_data()
    coadd_cat = Table.read(files_dict["coadd_cat"], format="fits")

    data = compute_data(
        sdss_data, coadd_cat, lims[model_name], sleep_net, binary_encoder, galaxy_encoder
    )
    return create_plotly_figure(data)


def create_plotly_figure(data):
    true, recon, res, true_locs, pred_locs = data
    images = np.stack((true, recon), axis=0)
    min_image = np.min(images)
    max_image = 1200
    fig = px.imshow(
        images,
        facet_col=0,
        facet_col_wrap=2,
        range_color=[min_image, max_image],
        color_continuous_scale=px.colors.sequential.Greys[::-1],
    )

    true_locs_scatter = go.Scatter(x=true_locs[:, 0], y=true_locs[:, 1], mode="markers")
    pred_locs_scatter = go.Scatter(x=pred_locs[:, 1], y=pred_locs[:, 0], mode="markers")

    # fig.add_trace(true_locs_scatter, row='all', col='all')
    # fig.add_trace(pred_locs_scatter, row='all', col='all')
    fig.add_trace(true_locs_scatter)
    fig.add_trace(pred_locs_scatter)

    return fig
