import matplotlib.pyplot as plt
import numpy as np
import torch
from einops import pack, rearrange, reduce
from matplotlib.figure import Figure
from torch import Tensor

from bliss.catalog import FullCatalog, TileCatalog
from bliss.datasets.background import add_noise_and_background
from bliss.datasets.generate_blends import render_full_catalog
from bliss.datasets.lsst import (
    convert_flux_to_mag,
    get_default_lsst_background,
    get_default_lsst_psf,
)
from bliss.encoders.autoencoder import CenteredGalaxyDecoder
from bliss.encoders.deblend import GalaxyEncoder
from bliss.encoders.detection import DetectionEncoder
from bliss.encoders.encoder import Encoder
from bliss.plotting import BlissFigure, plot_image
from bliss.render_tiles import (
    get_images_in_tiles,
    reconstruct_image_from_ptiles,
    render_galaxy_ptiles,
)


class ToySeparationFigure(BlissFigure):
    """Create figures related to assessingn probabilistic performance on toy blend."""

    @property
    def all_rcs(self):
        return {
            "three_separations": {
                "fontsize": 22,
                "tick_label_size": "small",
                "legend_fontsize": "small",
            },
            "toy_residuals": {"fontsize": 22},
            "toy_measurements": {
                "fontsize": 22,
                "tick_label_size": "small",
                "legend_fontsize": "small",
            },
        }

    @property
    def cache_name(self) -> str:
        return "toy_separation"

    @property
    def fignames(self) -> tuple[str, ...]:
        return ("three_separations", "toy_residuals", "toy_measurements")

    @property
    def separations_to_plot(self) -> list[int]:
        return [4, 8, 12]

    def compute_data(self, encoder: Encoder, decoder: CenteredGalaxyDecoder):
        # first, decide image size
        slen = 44
        bp = encoder.detection_encoder.bp
        tile_slen = encoder.detection_encoder.tile_slen
        size = slen + 2 * bp
        tile_slen = encoder.detection_encoder.tile_slen
        ptile_slen = encoder.detection_encoder.ptile_slen
        assert slen / tile_slen % 2 == 1, "Need odd number of tiles to center galaxy."

        # now separations between galaxies to be considered (in pixels)
        # for efficiency, we set the batch_size equal to the number of separations
        seps = torch.arange(0, 18, 0.1)
        batch_size = len(seps)

        # first centered galaxy, then moving one.
        colnames = (
            "fluxnorm_bulge",
            "fluxnorm_disk",
            "fluxnorm_agn",
            "a_b",
            "a_d",
            "b_b",
            "b_d",
            "pa_bulge",
            "i_ab",
            "flux",
        )
        assert len(colnames) == 10
        n_sources = 2
        flux1, flux2 = 5e6, 2.5e6
        mag1, mag2 = convert_flux_to_mag(torch.tensor([flux1, flux2]))
        mag1, mag2 = mag1.item(), mag2.item()
        gparam1 = [0, 1.0, 0, 0, 1.5, 0, 0.7, np.pi / 4, mag1, flux1]
        gparam2 = [0, 1.0, 0, 0, 1.0, 0, 0.7, 3 * np.pi / 4, mag2, flux2]
        gparams = torch.tensor([gparam1, gparam2])
        gparams = gparams.reshape(1, 2, 10).expand(batch_size, 2, 10)
        print(f"INFO: Fluxes correspond to magnitudes ({mag1},{mag2})")

        # need plocs for later
        x0, y0 = 22, 22  # center plocs
        plocs = torch.tensor([[[x0, y0], [x0, y0 + sep]] for sep in seps]).reshape(batch_size, 2, 2)

        psf = get_default_lsst_psf()
        bg = get_default_lsst_background()

        # create full catalogs (need separately since `render_blend`` only accepts 1 batch)
        images = torch.zeros(batch_size, 1, size, size)
        background = torch.full((batch_size, 1, size, size), bg)
        for ii in range(batch_size):
            plocs_ii = plocs[ii].reshape(1, 2, 2)
            d = {
                "n_sources": torch.full((1,), n_sources),
                "plocs": plocs_ii,
                "galaxy_bools": torch.ones(1, n_sources, 1),
                "galaxy_params": gparams[ii, None],
                "star_bools": torch.zeros(1, n_sources, 1),
                "star_fluxes": torch.zeros(1, n_sources, 1),
                "star_log_fluxes": torch.zeros(1, n_sources, 1),
            }
            full_cat = FullCatalog(slen, slen, d)
            image, _, _ = render_full_catalog(full_cat, psf, slen, bp)

            images[ii] = add_noise_and_background(image, background[ii])

        # predictions from encoder
        tile_est = encoder.variational_mode(images, background)

        # create reconstruction images and compute galaxy fluxes
        recon_ptiles = render_galaxy_ptiles(
            decoder,
            tile_est.locs,
            tile_est["galaxy_params"],
            tile_est["galaxy_bools"],
            ptile_slen,
            tile_slen,
            1,
        ).cpu()
        assert recon_ptiles.shape[-1] == recon_ptiles.shape[-2] == ptile_slen
        recon = reconstruct_image_from_ptiles(recon_ptiles, tile_slen)
        recon = recon.detach().cpu() + background

        # finally add flux
        # tile_est["fluxes"] = reduce(recon_ptiles, "b nth ntw c h w -> b nth ntw 1", "sum")
        residuals = (recon - images) / recon.sqrt()

        # now we need to obtain flux, pred. plocs, prob. of detection in tile and std. of plocs
        # for each source
        params = {
            "images": images,
            "recon": recon,
            "resid": residuals,
            "seps": seps,
            "truth": {
                "flux": torch.tensor([flux1, flux2]).reshape(1, 2, 1).expand(batch_size, 2, 1),
                "plocs": plocs,
            },
            "est": {
                "prob_n_source": torch.zeros(batch_size, 2, 1),
                "flux": torch.zeros(batch_size, 2, 1),
                "plocs": torch.zeros(batch_size, 2, 2),
                "plocs_sd": torch.zeros(batch_size, 2, 2),
            },
            "tile_est": tile_est.cpu().to_dict(),
        }

        # NOTE: in this case we don't want to zero out tiles with no source prediction!
        # manually extract source paramas as `encoder` automatically zeroes out sources
        tile_est = _get_source_params_not_zeroed(
            encoder.detection_encoder,
            encoder.galaxy_encoder,
            decoder,
            images.to(encoder.device),
            background.to(encoder.device),
        )

        for jj, sep in enumerate(seps):
            # get tile_est for a single batch
            d = tile_est.to_dict()
            d = {k: v[jj, None] for k, v in d.items()}
            tile_est_ii = TileCatalog(tile_slen, d)

            plocs_ii = plocs[jj]
            params_at_coord = tile_est_ii.get_tile_params_at_coord(plocs_ii)
            prob_n_source = params_at_coord["n_source_probs"]
            flux = params_at_coord["fluxes"]
            plocs_sd = params_at_coord["locs_sd"] * tile_slen
            locs = params_at_coord["locs"]
            assert prob_n_source.shape == flux.shape == (2, 1)
            assert plocs_sd.shape == locs.shape == (2, 2)

            if sep < 2:
                params["est"]["prob_n_source"][jj][0] = prob_n_source[0]
                params["est"]["flux"][jj][0] = flux[0]
                params["est"]["plocs"][jj][0] = locs[0] * tile_slen + 5 * tile_slen
                params["est"]["plocs_sd"][jj][0] = plocs_sd[0]

                params["est"]["prob_n_source"][jj][1] = torch.nan
                params["est"]["flux"][jj][1] = torch.nan
                params["est"]["plocs"][jj][1] = torch.tensor([torch.nan, torch.nan])
                params["est"]["plocs_sd"][jj][1] = torch.tensor([torch.nan, torch.nan])
            else:
                bias = 5 + np.ceil((sep - 2) / 4)
                params["est"]["prob_n_source"][jj] = prob_n_source
                params["est"]["flux"][jj] = flux
                params["est"]["plocs"][jj][0] = locs[0] * tile_slen + 5 * tile_slen
                params["est"]["plocs"][jj, 1, 0] = locs[1][0] * tile_slen + 5 * tile_slen
                params["est"]["plocs"][jj, 1, 1] = locs[1][1] * tile_slen + bias * tile_slen
                params["est"]["plocs_sd"][jj] = plocs_sd

        return params

    def _get_three_separations_plot(self, data) -> Figure:
        seps: np.ndarray = data["seps"]
        images: np.ndarray = data["images"]
        tplocs: np.ndarray = data["truth"]["plocs"]
        eplocs: np.ndarray = data["est"]["plocs"]

        # first, create image with 3 example separations (very blended to not blended)
        bp = 24
        fig, axes = plt.subplots(1, 3, figsize=(12, 7))
        axes = axes.flatten()
        seps_to_plot = self.separations_to_plot
        trim = 25  # zoom into relevant part of the image

        c1 = plt.rcParams["axes.prop_cycle"].by_key()["color"][1]  # true
        c2 = plt.rcParams["axes.prop_cycle"].by_key()["color"][3]  # predicted

        for ii, psep in enumerate(seps_to_plot):
            indx = list(seps).index(psep)
            image = images[indx, 0, trim:-trim, trim:-trim]
            x1 = tplocs[indx, :, 1] + bp - 0.5 - trim
            y1 = tplocs[indx, :, 0] + bp - 0.5 - trim
            x2 = eplocs[indx, :, 1] + bp - 0.5 - trim
            y2 = eplocs[indx, :, 0] + bp - 0.5 - trim
            axes[ii].imshow(image)
            axes[ii].scatter(x1, y1, marker="x", color="r", s=30, label=None if ii else "Truth")
            axes[ii].scatter(x2, y2, marker="+", color="b", s=50, label=None if ii else "Predicted")
            axes[ii].set_xticks([0, 10, 20, 30, 40])
            axes[ii].set_yticks([0, 10, 20, 30, 40])
            axes[ii].set_title(rf"\rm Separation: {psep} pixels")

            if ii == 0:
                axes[ii].legend(loc="best", prop={"size": 14}, markerscale=2)

            if ii > 0:
                axes[ii].set_yticks([])  # turn off axis
                axes[ii].set_ylim(axes[0].get_ylim())  # align axes

            axes[ii].text(x1[0].item(), y1[0].item() - 7, "1", color=c1)
            axes[ii].text(x1[1].item(), y1[1].item() - 7, "2", color=c2)

        fig.tight_layout()

        return fig

    def _get_residuals_figure(self, data) -> Figure:
        n_examples = 3
        bp = 24
        seps_to_plot = [4, 8, 12]
        seps = data["seps"]
        tplocs: np.ndarray = data["truth"]["plocs"]
        images_all, recon_all, res_all = data["images"], data["recon"], data["resid"]
        trim = 20
        indices = np.array([list(seps).index(psep) for psep in seps_to_plot]).astype(int)

        images = images_all[indices, 0, trim + 10 : -trim - 10, trim + 15 : -trim - 5]
        recons = recon_all[indices, 0, trim + 10 : -trim - 10, trim + 15 : -trim - 5]
        residuals = res_all[indices, 0, trim + 10 : -trim - 10, trim + 15 : -trim - 5]
        x1 = tplocs[indices, :, 1] + bp - 0.5 - trim - 15
        y1 = tplocs[indices, :, 0] + bp - 0.5 - trim - 10

        pad = 6.0
        fig, axes = plt.subplots(nrows=n_examples, ncols=3, figsize=(11, 18))

        for i in range(n_examples):
            ax_true = axes[i, 0]
            ax_recon = axes[i, 1]
            ax_res = axes[i, 2]

            # only add titles to the first axes.
            if i == 0:
                ax_true.set_title(r"\rm Images $x$", pad=pad)
                ax_recon.set_title(r"\rm Reconstruction $\tilde{x}$", pad=pad)
                ax_res.set_title(
                    r"Residual $\left(\tilde{x} - x\right) / \sqrt{\tilde{x}}$", pad=pad
                )

            ax_true.scatter(x1[i], y1[i], color="r", alpha=0.5, s=40, marker="x", label="Truth")
            ax_recon.scatter(x1[i], y1[i], color="r", alpha=0.5, s=40, marker="x")
            ax_res.scatter(x1[i], y1[i], color="r", alpha=0.5, s=40, marker="x")

            # standarize ranges of true and reconstruction
            image = images[i]
            recon = recons[i]
            res = residuals[i]

            vmin = min(image.min().item(), recon.min().item())
            vmax = max(image.max().item(), recon.max().item())
            vmin_res = res.min().item()
            vmax_res = res.max().item()
            vres = max(abs(vmin_res), abs(vmax_res))

            # plot images
            plot_image(fig, ax_true, image, vrange=(vmin, vmax))
            plot_image(fig, ax_recon, recon, vrange=(vmin, vmax))
            plot_image(fig, ax_res, res, vrange=(-vres, vres), cmap="bwr")

            ax_true.set_xticks([0, 10, 20, 30])
            ax_true.set_yticks([0, 10, 20, 30])
            ax_recon.set_xticks([0, 10, 20, 30])
            ax_recon.set_yticks([0, 10, 20, 30])
            ax_res.set_xticks([0, 10, 20, 30])
            ax_res.set_yticks([0, 10, 20, 30])

            if i == 0:
                ax_true.legend(loc="best", prop={"size": 14}, markerscale=2)

        plt.subplots_adjust(hspace=-0.9)
        plt.tight_layout()
        return fig

    def _get_measurement_figure(self, data: dict) -> Figure:
        fig, axes = plt.subplots(2, 2, figsize=(12, 12))
        axs = axes.flatten()
        seps = data["seps"]
        xticks = [sep for sep in seps if sep % 2 == 0]

        c1 = plt.rcParams["axes.prop_cycle"].by_key()["color"][1]
        c2 = plt.rcParams["axes.prop_cycle"].by_key()["color"][3]

        # probability of detection in each tile
        prob_n1 = data["est"]["prob_n_source"][:, 0]
        prob_n2 = data["est"]["prob_n_source"][:, 1]

        axs[0].plot(seps, prob_n1, "-", label="Galaxy 1", color=c1)
        axs[0].plot(seps, prob_n2, "-", label="Galaxy 2", color=c2)
        axs[0].axvline(2, color="k", ls="--", label=r"\rm Tile boundary", alpha=0.5)
        axs[0].axvline(6, ls="--", color="k", alpha=0.5)
        axs[0].axvline(10, ls="--", color="k", alpha=0.5)
        axs[0].axvline(14, ls="--", color="k", alpha=0.5)
        axs[0].legend(loc="best")
        axs[0].set_xticks(xticks)
        axs[0].set_xlim(0, 16)
        axs[0].set_xlabel(r"\rm Separation (pixels)")
        axs[0].set_ylabel(r"\rm Detection Probability")

        # distance residual
        tploc1 = data["truth"]["plocs"][:, 0]
        tploc2 = data["truth"]["plocs"][:, 1]
        eploc1 = data["est"]["plocs"][:, 0]
        eploc2 = data["est"]["plocs"][:, 1]
        dist1 = ((tploc1 - eploc1) ** 2).sum(1) ** (1 / 2)
        dist2 = ((tploc2 - eploc2) ** 2).sum(1) ** (1 / 2)

        axs[1].plot(seps, dist1, "-", color=c1)
        axs[1].plot(seps, dist2, "-", color=c2)
        axs[1].axhline(0, color="k", ls="-", alpha=1.0)
        axs[1].axvline(2, ls="--", color="k", label=r"\rm Tile boundary", alpha=0.5)
        axs[1].axvline(6, ls="--", color="k", alpha=0.5)
        axs[1].axvline(10, ls="--", color="k", alpha=0.5)
        axs[1].axvline(14, ls="--", color="k", alpha=0.5)
        axs[1].set_xticks(xticks)
        axs[1].set_xlim(0, 16)
        axs[1].set_xlabel(r"\rm Separation (pixels)")
        axs[1].set_ylabel(r"\rm Centroid location residual (pixels)")

        # location error (squared sum) estimate
        eploc_sd1 = (data["est"]["plocs_sd"][:, 0] ** 2).sum(1) ** (1 / 2)
        eploc_sd2 = (data["est"]["plocs_sd"][:, 1] ** 2).sum(1) ** (1 / 2)
        axs[3].plot(seps, eploc_sd1, "-", color=c1)
        axs[3].plot(seps, eploc_sd2, "-", color=c2)
        axs[3].axhline(0, color="k", ls="-", alpha=1.0)
        axs[3].axvline(2, ls="--", color="k", label=r"\rm Tile boundary", alpha=0.5)
        axs[3].axvline(6, ls="--", color="k", alpha=0.5)
        axs[3].axvline(10, ls="--", color="k", alpha=0.5)
        axs[3].axvline(14, ls="--", color="k", alpha=0.5)
        axs[3].set_xticks(xticks)
        axs[3].set_xlim(0, 16)
        axs[3].set_ylim(axs[1].get_ylim())
        axs[3].set_xlabel(r"\rm Separation (pixels)")
        axs[3].set_ylabel(r"\rm Predicted centroid std. (pixels)")

        # flux normalized residuals
        tflux1 = data["truth"]["flux"][:, 0]
        tflux2 = data["truth"]["flux"][:, 1]
        eflux1 = data["est"]["flux"][:, 0]
        eflux2 = data["est"]["flux"][:, 1]
        rflux1 = (eflux1 - tflux1) / tflux1
        rflux2 = (eflux2 - tflux2) / tflux2

        axs[2].plot(seps, rflux1, "-", color=c1)
        axs[2].plot(seps, rflux2, "-", color=c2)
        axs[2].axhline(0, color="k", ls="-", alpha=1.0)
        axs[2].axvline(2, ls="--", color="k", label=r"\rm Tile boundary", alpha=0.5)
        axs[2].axvline(6, ls="--", color="k", alpha=0.5)
        axs[2].axvline(10, ls="--", color="k", alpha=0.5)
        axs[2].axvline(14, ls="--", color="k", alpha=0.5)
        axs[2].set_xticks(xticks)
        axs[2].set_xlim(0, 16)
        axs[2].set_xlabel(r"\rm Separation (pixels)")
        axs[2].set_ylabel(r"\rm $(f^{\rm recon} - f^{\rm true}) / f^{\rm true}$")

        return fig

    def create_figure(self, fname: str, data: dict) -> Figure:
        if fname == "three_separations":
            return self._get_three_separations_plot(data)
        if fname == "toy_residuals":
            return self._get_residuals_figure(data)
        if fname == "toy_measurements":
            return self._get_measurement_figure(data)
        raise NotImplementedError("Figure {fname} not implemented.")


def _get_source_params_not_zeroed(
    detection_encoder: DetectionEncoder,
    galaxy_encoder: GalaxyEncoder,
    decoder: CenteredGalaxyDecoder,
    images: Tensor,
    background: Tensor,
) -> TileCatalog:
    tiled_params: dict[str, Tensor] = {}

    # should all fit in GPU at once
    img_bg, _ = pack([images, background], "b * h w")  # by default concatenate channels
    ptiles = get_images_in_tiles(img_bg, detection_encoder.tile_slen, detection_encoder.ptile_slen)
    ptiles_flat = rearrange(ptiles, "b nth ntw c h w -> (b nth ntw) c h w")

    n_source_probs, locs_mean, locs_sd_raw = detection_encoder.encode_tiled(ptiles_flat)
    n_sources = n_source_probs.ge(0.5).long()
    n_source_probs_inflated = rearrange(n_source_probs, "n -> n 1")  # for `TileCatalog`

    tiled_params.update({"n_sources": n_sources, "n_source_probs": n_source_probs_inflated})
    tiled_params.update({"locs": locs_mean, "locs_sd": locs_sd_raw})

    galaxy_params = galaxy_encoder.forward(ptiles_flat, locs_mean)
    tiled_params.update({"galaxy_params": galaxy_params})

    # get tile catalog
    n_tiles_h = (images.shape[2] - 2 * detection_encoder.bp) // detection_encoder.tile_slen
    n_tiles_w = (images.shape[3] - 2 * detection_encoder.bp) // detection_encoder.tile_slen
    tiled_est = TileCatalog.from_flat_dict(
        detection_encoder.tile_slen,
        n_tiles_h,
        n_tiles_w,
        {k: v.squeeze(0) for k, v in tiled_params.items()},
    )

    # now get flux
    recon_ptiles = render_galaxy_ptiles(
        decoder,
        tiled_est.locs,
        tiled_est["galaxy_params"],
        torch.ones_like(tiled_est["n_source_probs"]),
        detection_encoder.ptile_slen,
        detection_encoder.tile_slen,
        1,
    )
    fluxes = reduce(recon_ptiles, "b nth ntw c h w -> b nth ntw 1", "sum")
    tiled_est["fluxes"] = fluxes
    return tiled_est.to(torch.device("cpu"))
