import argparse
from pathlib import Path

import torch
import numpy as np
from matplotlib import pyplot as plt
from astropy.table import Table

from bliss.datasets.sdss import SloanDigitalSkySurvey
from bliss.models.binary import BinaryEncoder
from bliss.models.galaxy_encoder import GalaxyEncoder
from bliss.encoder import Encoder
from bliss.reconstruct import reconstruct_scene_at_coordinates
from bliss.sleep import SleepPhase
from bliss import reporting


def get_sdss_data(sdss_pixel_scale=0.396):
    run = 94
    camcol = 1
    field = 12
    bands = (2,)
    sdss_data = SloanDigitalSkySurvey(
        sdss_dir="data/sdss",
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


scenes = {
    # "sdss_recon0": ((1700, 2000), (200, 500)),  # scene
    # "sdss_recon1": ((1000, 1300), (1150, 1450)),  # scene
    # "sdss_recon2": ((742, 790), (460, 508)),  # individual blend
    # "sdss_recon3": ((1128, 1160), (25, 57)),  # individual blend
    # "sdss_recon4": ((500, 552), (170, 202)),  # individual blend
    "sdss_recon0": (200, 1700),
    "sdss_recon1": (1150, 1000),
}
SCENE_SIZE = 300


def create_figure(true, recon, res, locs_true=None, map_recon=None):
    """Make figures related to detection and classification in SDSS."""
    plt.style.use("seaborn-colorblind")
    pad = 6.0
    reporting.set_rc_params(fontsize=22, tick_label_size="small", legend_fontsize="small")
    # for figname in self.fignames:
    # true, recon, res, locs, locs_pred = data
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

    if locs_true is not None:
        ax_true.scatter(locs_true[:, 1], locs_true[:, 0], color="g", marker="+", s=20)
        ax_recon.scatter(
            locs_true[:, 1], locs_true[:, 0], color="g", marker="+", s=20, label="SDSS Object"
        )
        ax_res.scatter(locs_true[:, 1], locs_true[:, 0], color="g", marker="+", s=20)

    if map_recon is not None:
        locs_pred = map_recon["plocs"][0]
        star_bool = map_recon["star_bool"][0]
        galaxy_bool = map_recon["galaxy_bool"][0]
        locs_galaxies = locs_pred[galaxy_bool[:, 0] > 0.5, :]
        locs_stars = locs_pred[star_bool[:, 0] > 0.5, :]
        if locs_galaxies.shape[0] > 0:
            ax_true.scatter(locs_galaxies[:, 1], locs_galaxies[:, 0], color="c", marker="x", s=20)
            ax_recon.scatter(
                locs_galaxies[:, 1],
                locs_galaxies[:, 0],
                color="c",
                marker="x",
                s=20,
                label="Predicted Galaxy",
            )
            ax_res.scatter(locs_galaxies[:, 1], locs_galaxies[:, 0], color="c", marker="x", s=20)
        if locs_stars.shape[0] > 0:
            ax_true.scatter(locs_stars[:, 1], locs_stars[:, 0], color="r", marker="x", s=20)
            ax_recon.scatter(
                locs_stars[:, 1],
                locs_stars[:, 0],
                color="r",
                marker="x",
                s=20,
                label="Predicted Star",
            )
            ax_res.scatter(locs_stars[:, 1], locs_stars[:, 0], color="r", marker="x", s=20)
        ax_recon.legend(
            bbox_to_anchor=(0.0, 1.2, 1.0, 0.102),
            loc="lower left",
            ncol=2,
            mode="expand",
            borderaxespad=0.0,
        )

    plt.subplots_adjust(hspace=-0.4)
    plt.tight_layout()

    return fig


def get_true_locs_from_coadd(coadd_cat, h, w, scene_size):
    locs_true_w = torch.from_numpy(np.array(coadd_cat["x"]).astype(float))
    locs_true_h = torch.from_numpy(np.array(coadd_cat["y"]).astype(float))
    locs_true = torch.stack((locs_true_h, locs_true_w), dim=1)
    locs_true = locs_true[h < locs_true[:, 0]]
    locs_true = locs_true[w < locs_true[:, 1]]

    locs_true = locs_true[locs_true[:, 0] < h + scene_size]
    locs_true = locs_true[locs_true[:, 1] < w + scene_size]

    # Shift locs by lower limit
    locs_true[:, 0] = locs_true[:, 0] - h
    locs_true[:, 1] = locs_true[:, 1] - w

    return locs_true


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create figures related to SDSS galaxies.")
    parser.add_argument(
        "-o",
        "--output",
        default="output/sdss_figures",
        type=str,
        help="Where to save figures and caches relative to $BLISS_HOME.",
    )
    args = vars(parser.parse_args())
    sdss_data = get_sdss_data()
    my_image = torch.from_numpy(sdss_data["image"]).unsqueeze(0).unsqueeze(0)
    coadd_cat = Table.read("data/coadd_catalog_94_1_12.fits", format="fits")

    sleep = SleepPhase.load_from_checkpoint("models/sdss_sleep.ckpt").to("cuda").eval()
    binary = BinaryEncoder.load_from_checkpoint("models/sdss_binary.ckpt").to("cuda").eval()
    galaxy = GalaxyEncoder.load_from_checkpoint("models/sdss_galaxy_encoder.ckpt").to("cuda").eval()
    location = sleep.image_encoder.to("cuda").eval()
    dec = sleep.image_decoder.to("cuda").eval()
    encoder = Encoder(location.eval(), binary.eval(), galaxy.eval()).to("cuda")

    outdir = Path(args["output"])
    outdir.mkdir(exist_ok=True)

    for scene_name, scene_coords in scenes.items():
        h, w = scene_coords
        true = my_image[:, :, h : (h + SCENE_SIZE), w : (w + SCENE_SIZE)]
        locs_true = get_true_locs_from_coadd(coadd_cat, h, w, SCENE_SIZE)
        recon, map_recon = reconstruct_scene_at_coordinates(
            encoder, dec, my_image, h, w, SCENE_SIZE
        )
        resid = (true - recon) / recon.sqrt()
        fig = create_figure(
            true[0, 0], recon[0, 0], resid[0, 0], locs_true=locs_true, map_recon=map_recon
        )
        fig.savefig(outdir / (scene_name + ".pdf"), format="pdf")
