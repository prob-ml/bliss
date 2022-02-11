# flake8: noqa
# pylint: skip-file
import argparse
from pathlib import Path

import numpy as np
import torch
from astropy.table import Table
from hydra.utils import instantiate
from matplotlib import pyplot as plt

from bliss import reporting
from bliss.datasets.sdss import SloanDigitalSkySurvey
from bliss.encoder import Encoder
from bliss.inference import reconstruct_scene_at_coordinates
from bliss.models.binary import BinaryEncoder
from bliss.models.galaxy_encoder import GalaxyEncoder
from bliss.sleep import SleepPhase


def reconstruct(cfg):
    sdss_data = get_sdss_data(cfg.paths.sdss, cfg.reconstruct.sdss_pixel_scale)
    my_image = torch.from_numpy(sdss_data["image"]).unsqueeze(0).unsqueeze(0)
    coadd_cat = Table.read(cfg.reconstruct.coadd_cat, format="fits")
    device = torch.device(cfg.reconstruct.device)

    sleep = instantiate(cfg.models.sleep).to(device).eval()
    sleep.load_state_dict(torch.load(cfg.predict.sleep_checkpoint, map_location=sleep.device))

    binary = instantiate(cfg.models.binary).to(device).eval()
    binary.load_state_dict(torch.load(cfg.predict.binary_checkpoint, map_location=binary.device))

    galaxy = instantiate(cfg.models.galaxy_encoder).to(device).eval()
    if cfg.reconstruct.real:
        galaxy_state_dict = torch.load(
            cfg.predict.galaxy_checkpoint_real, map_location=galaxy.device
        )
    else:
        galaxy_state_dict = torch.load(cfg.predict.galaxy_checkpoint, map_location=galaxy.device)
    galaxy.load_state_dict(galaxy_state_dict)
    location = sleep.image_encoder.to(device).eval()
    dec = sleep.image_decoder.to(device).eval()
    encoder = Encoder(location.eval(), binary.eval(), galaxy.eval()).to(device)

    if cfg.reconstruct.outdir is not None:
        outdir = Path(cfg.reconstruct.outdir)
        outdir.mkdir(exist_ok=True)
    else:
        outdir = None

    for scene_name, scene_coords in cfg.reconstruct.scenes.items():
        h, w, scene_size = scene_coords["h"], scene_coords["w"], scene_coords["size"]
        true = my_image[:, :, h : (h + scene_size), w : (w + scene_size)]
        coadd_objects = get_objects_from_coadd(coadd_cat, h, w, scene_size)
        recon, map_recon = reconstruct_scene_at_coordinates(
            encoder, dec, my_image, h, w, scene_size, device=device
        )
        resid = (true - recon) / recon.sqrt()
        if outdir is not None:
            fig = create_figure(
                true[0, 0],
                recon[0, 0],
                resid[0, 0],
                coadd_objects=coadd_objects,
                map_recon=map_recon,
            )
            fig.savefig(outdir / (scene_name + ".pdf"), format="pdf")


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
        "wcs": sdss_data[0]["wcs"][0],
        "pixel_scale": sdss_pixel_scale,
    }


def create_figure(true, recon, res, coadd_objects=None, map_recon=None):
    """Make figures related to detection and classification in SDSS."""
    plt.style.use("seaborn-colorblind")
    pad = 6.0
    reporting.set_rc_params(fontsize=22, tick_label_size="small", legend_fontsize="small")
    # for figname in self.fignames:
    # true, recon, res, locs, locs_pred = data
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(28, 12))
    assert len(true.shape) == len(recon.shape) == len(res.shape) == 2

    # pick standard ranges for residuals
    scene_size = true.shape[-1]
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

    if coadd_objects is not None:
        locs_true = coadd_objects["locs"]
        galaxy_bool_true = coadd_objects["galaxy_bool"]
        locs_galaxies_true = locs_true[galaxy_bool_true > 0.5]
        locs_stars_true = locs_true[galaxy_bool_true < 0.5]
        if locs_stars_true.shape[0] > 0:
            ax_true.scatter(
                locs_stars_true[:, 1], locs_stars_true[:, 0], color="b", marker="+", s=20
            )
            ax_recon.scatter(
                locs_stars_true[:, 1],
                locs_stars_true[:, 0],
                color="b",
                marker="+",
                s=20,
                label="SDSS Stars",
            )
            ax_res.scatter(
                locs_galaxies_true[:, 1], locs_galaxies_true[:, 0], color="b", marker="+", s=20
            )
        if locs_galaxies_true.shape[0] > 0:
            ax_true.scatter(
                locs_galaxies_true[:, 1], locs_galaxies_true[:, 0], color="m", marker="+", s=20
            )
            ax_recon.scatter(
                locs_galaxies_true[:, 1],
                locs_galaxies_true[:, 0],
                color="m",
                marker="+",
                s=20,
                label="SDSS Galaxies",
            )
            ax_res.scatter(
                locs_galaxies_true[:, 1], locs_galaxies_true[:, 0], color="m", marker="+", s=20
            )

    if map_recon is not None:
        locs_pred = map_recon["plocs"][0]
        star_bool = map_recon["star_bool"][0]
        galaxy_bool = map_recon["galaxy_bool"][0]
        locs_galaxies = locs_pred[galaxy_bool[:, 0] > 0.5, :]
        locs_stars = locs_pred[star_bool[:, 0] > 0.5, :]
        if locs_galaxies.shape[0] > 0:
            in_bounds = torch.all((locs_galaxies > 0) & (locs_galaxies < scene_size), dim=-1)
            locs_galaxies = locs_galaxies[in_bounds]
            ax_true.scatter(locs_galaxies[:, 1], locs_galaxies[:, 0], color="c", marker="x", s=20)
            ax_recon.scatter(
                locs_galaxies[:, 1],
                locs_galaxies[:, 0],
                color="c",
                marker="x",
                s=20,
                label="Predicted Galaxy",
                alpha=0.6,
            )
            ax_res.scatter(locs_galaxies[:, 1], locs_galaxies[:, 0], color="c", marker="x", s=20)
        if locs_stars.shape[0] > 0:
            in_bounds = torch.all((locs_stars > 0) & (locs_stars < scene_size), dim=-1)
            locs_stars = locs_stars[in_bounds]
            ax_true.scatter(
                locs_stars[:, 1], locs_stars[:, 0], color="r", marker="x", s=20, alpha=0.6
            )
            ax_recon.scatter(
                locs_stars[:, 1],
                locs_stars[:, 0],
                color="r",
                marker="x",
                s=20,
                label="Predicted Star",
                alpha=0.6,
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


def get_objects_from_coadd(coadd_cat, h, w, scene_size):
    locs_true_w = torch.from_numpy(np.array(coadd_cat["x"]).astype(float))
    locs_true_h = torch.from_numpy(np.array(coadd_cat["y"]).astype(float))
    locs_true = torch.stack((locs_true_h, locs_true_w), dim=1)

    ## Get relevant objects
    objects_in_scene = (
        (h < locs_true[:, 0])
        & (w < locs_true[:, 1])
        & (h + scene_size > locs_true[:, 0])
        & (w + scene_size > locs_true[:, 1])
    )
    locs_true = locs_true[objects_in_scene]
    galaxy_bool = torch.from_numpy(np.array(coadd_cat["galaxy_bool"]).astype(float))
    galaxy_bool = galaxy_bool[objects_in_scene]

    # Shift locs by lower limit
    locs_true[:, 0] = locs_true[:, 0] - h
    locs_true[:, 1] = locs_true[:, 1] - w

    return {
        "locs": locs_true,
        "galaxy_bool": galaxy_bool,
    }


if __name__ == "__main__":
    pass
    # parser = argparse.ArgumentParser(description="Create figures related to SDSS galaxies.")
    # parser.add_argument(
    #     "-o",
    #     "--output",
    #     default="output/sdss_figures",
    #     type=str,
    #     help="Where to save figures and caches relative to $BLISS_HOME.",
    # )
    # parser.add_argument(
    #     "-r",
    #     "--real",
    #     action="store_true",
    #     help="Use galaxy encoder trained on real images",
    # )
    # args = vars(parser.parse_args())
