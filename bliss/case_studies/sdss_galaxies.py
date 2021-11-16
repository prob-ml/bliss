from typing import Tuple
import torch
import numpy as np
from matplotlib import pyplot as plt

from bliss.datasets import sdss
from bliss import reporting
from bliss.models.binary import BinaryEncoder
from bliss.models.galaxy_encoder import GalaxyEncoder
from bliss.sleep import SleepPhase
from bliss.predict import predict_on_image


# class SDSSReconstructionFigures:


def compute_data(
    scene: Tuple[torch.Tensor, np.ndarray],
    lims: Tuple[int, int],
    sleep_net: SleepPhase,
    binary_encoder: BinaryEncoder,
    galaxy_encoder: GalaxyEncoder,
):
    # scene = get_sdss_data()["image"]
    # sdss_rec_fig = SDSSReconstructionFigures(outdir, overwrite=overwrite)
    # sdss_rec_fig.save_figures(scene, sleep_net, binary_encoder, galaxy_encoder)
    assert isinstance(scene, (torch.Tensor, np.ndarray))
    assert sleep_net.device == binary_encoder.device == galaxy_encoder.device
    device = sleep_net.device

    bp = 24

    image_encoder = sleep_net.image_encoder.to(device).eval()
    image_decoder = sleep_net.image_decoder.to(device).eval()

    bp = image_encoder.border_padding
    # for figname in self.fignames:
    #     xlim, ylim = self.lims[figname]
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

        tile_map, _, _ = predict_on_image(chunk, image_encoder, binary_encoder, galaxy_encoder)

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

    data = (true_image, recon_image, residual)

    return data


def create_figure(data):
    """Make figures related to detection and classification in SDSS."""
    plt.style.use("seaborn-colorblind")
    pad = 6.0
    reporting.set_rc_params(fontsize=22, tick_label_size="small", legend_fontsize="small")
    # for figname in self.fignames:
    true, recon, res = data
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

    plt.subplots_adjust(hspace=-0.4)
    plt.tight_layout()

    return fig


def get_sdss_data():
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
        "pixel_scale": SDSS_PIXEL_SCALE,
    }
