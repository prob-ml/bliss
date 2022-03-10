# flake8: noqa
# pylint: skip-file
from pathlib import Path

import torch
from astropy.table import Table
from hydra.utils import instantiate
from matplotlib import pyplot as plt

from bliss import reporting
from bliss.datasets.sdss import SloanDigitalSkySurvey, convert_flux_to_mag
from bliss.encoder import Encoder
from bliss.inference import infer_blends, reconstruct_scene_at_coordinates
from bliss.models.location_encoder import get_full_params_from_tiles
from case_studies.sdss_galaxies.plots import set_rc_params


def reconstruct(cfg):
    sdss_data = get_sdss_data(cfg.paths.sdss, cfg.reconstruct.sdss_pixel_scale)
    my_image = torch.from_numpy(sdss_data["image"]).unsqueeze(0).unsqueeze(0)
    my_background = torch.from_numpy(sdss_data["background"]).unsqueeze(0).unsqueeze(0)
    # Apply masks
    my_image[:, :, 1200:1360, 1700:1900] = 865.0 + (
        torch.tensor(865.0).sqrt() * torch.randn_like(my_image[:, :, 1200:1360, 1700:1900])
    )
    my_image[:, :, 280:400, 1220:1320] = 865.0 + (
        torch.tensor(865.0).sqrt() * torch.randn_like(my_image[:, :, 280:400, 1220:1320])
    )
    my_background[:, :, 1200:1360, 1700:1900] = 865.0
    my_background[:, :, 280:400, 1220:1320] = 865.0
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
        bp = encoder.border_padding
        h, w, scene_size = scene_coords["h"], scene_coords["w"], scene_coords["size"]
        if scene_size == "all":
            h = bp
            w = bp
            h_end = ((my_image.shape[2] - 2 * bp) // 4) * 4 + bp
            w_end = ((my_image.shape[3] - 2 * bp) // 4) * 4 + bp
        else:
            h_end = h + scene_size
            w_end = w + scene_size
        true = my_image[:, :, h:h_end, w:w_end]
        coadd_data = reporting.get_params_from_coadd(
            coadd_cat,
            xlim=(w, w_end),
            ylim=(h, h_end),
            shift_plocs_to_lim_start=True,
            convert_xy_to_hw=True,
        )
        recon, tile_map_recon = reconstruct_scene_at_coordinates(
            encoder,
            dec,
            my_image,
            my_background,
            (h, h_end),
            (w, w_end),
            slen=cfg.reconstruct.slen,
            device=device,
        )
        tile_map_recon["galaxy_blends"] = infer_blends(tile_map_recon, 2)
        print(
            f"{(tile_map_recon['galaxy_blends'] > 1).sum()} galaxies are part of blends in image."
        )
        map_recon = get_full_params_from_tiles(tile_map_recon, encoder.tile_slen)
        map_recon["fluxes"] = (
            map_recon["galaxy_bools"] * map_recon["galaxy_fluxes"]
            + map_recon["star_bools"] * map_recon["fluxes"]
        )
        map_recon["mags"] = convert_flux_to_mag(map_recon["fluxes"])
        map_recon["plocs"] = map_recon["plocs"] - 0.5
        scene_metrics_map = {}
        for mag_cut in (20, 24):
            scene_metrics_map[mag_cut] = reporting.scene_metrics(
                coadd_data,
                map_recon,
                mag_cut=mag_cut,
            )
        resid = (true - recon) / recon.sqrt()
        if outdir is not None:
            fig = create_figure(
                true[0, 0],
                recon[0, 0],
                resid[0, 0],
                coadd_objects=coadd_data,
                map_recon=map_recon,
            )
            fig.savefig(outdir / (scene_name + ".pdf"), format="pdf")
            torch.save(scene_metrics_map, outdir / (scene_name + ".pt"))
            torch.save(coadd_data, outdir / (scene_name + "_coadd.pt"))
            torch.save(map_recon, outdir / (scene_name + "_map_recon.pt"))


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
        "background": sdss_data[0]["background"][0],
        "wcs": sdss_data[0]["wcs"][0],
        "pixel_scale": sdss_pixel_scale,
    }


def create_figure(true, recon, res, coadd_objects=None, map_recon=None):
    """Make figures related to detection and classification in SDSS."""
    plt.style.use("seaborn-colorblind")
    pad = 6.0
    set_rc_params(fontsize=22, tick_label_size="small", legend_fontsize="small")
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(28, 12))
    assert len(true.shape) == len(recon.shape) == len(res.shape) == 2

    # pick standard ranges for residuals
    scene_size = max(true.shape[-2], true.shape[-1])
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
        locs_true = coadd_objects["plocs"]
        true_galaxy_bools = coadd_objects["galaxy_bools"]
        locs_galaxies_true = locs_true[true_galaxy_bools > 0.5]
        locs_stars_true = locs_true[true_galaxy_bools < 0.5]
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
        star_bools = map_recon["star_bools"][0]
        galaxy_bools = map_recon["galaxy_bools"][0]
        locs_galaxies = locs_pred[galaxy_bools[:, 0] > 0.5, :]
        locs_stars = locs_pred[star_bools[:, 0] > 0.5, :]
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


if __name__ == "__main__":
    pass
