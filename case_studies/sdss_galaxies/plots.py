#!/usr/bin/env python3
"""Produce all figures. Save to nice PNG format."""
import warnings
from abc import abstractmethod
from pathlib import Path

import galsim
import hydra
import matplotlib as mpl
import numpy as np
import pytorch_lightning as pl
import seaborn as sns
import torch
from astropy.table import Table
from astropy.wcs.wcs import WCS
from hydra.utils import instantiate
from matplotlib import pyplot as plt

from bliss import generate, reporting
from bliss.datasets import sdss
from bliss.datasets.galsim_galaxies import load_psf_from_file
from bliss.encoder import Encoder
from bliss.inference import reconstruct_scene_at_coordinates
from bliss.models.galaxy_net import OneCenteredGalaxyAE

pl.seed_everything(0)


def remove_outliers(*args, level=0.99):
    # each arg in args should be 1D numpy.array with same number of data points.
    for arg in args:
        assert len(arg) == len(args[0])
        assert isinstance(arg, np.ndarray)
        assert len(arg.shape) == 1
    keep = np.ones(args[0].shape).astype(bool)
    for x in args:
        x_min, x_max = np.quantile(x, 1 - level), np.quantile(x, level)
        keep_x = (x > x_min) & (x < x_max)
        keep &= keep_x

    return (arg[keep] for arg in args)


def get_sdss_data(cfg):
    run = 94
    camcol = 1
    field = 12
    bands = (2,)
    sdss_data = sdss.SloanDigitalSkySurvey(
        sdss_dir=cfg.paths.sdss,
        run=run,
        camcol=camcol,
        fields=(field,),
        bands=bands,
    )

    return {
        "image": sdss_data[0]["image"][0],
        "background": sdss_data[0]["background"][0],
        "wcs": sdss_data[0]["wcs"][0],
        "pixel_scale": cfg.plots.sdss_pixel_scale,
    }


def add_extra_coadd_info(coadd_cat_file: str, psf_image_file: str, pixel_scale: float, wcs: WCS):
    """Add additional useful information to coadd catalog."""
    coadd_cat = Table.read(coadd_cat_file)

    psf = load_psf_from_file(psf_image_file, pixel_scale)
    x, y = wcs.all_world2pix(coadd_cat["ra"], coadd_cat["dec"], 0)
    galaxy_bools = ~coadd_cat["probpsf"].data.astype(bool)
    flux, mag = reporting.get_flux_coadd(coadd_cat)
    hlr = reporting.get_hlr_coadd(coadd_cat, psf)

    coadd_cat["x"] = x
    coadd_cat["y"] = y
    coadd_cat["galaxy_bool"] = galaxy_bools
    coadd_cat["flux"] = flux
    coadd_cat["mag"] = mag
    coadd_cat["hlr"] = hlr
    coadd_cat.replace_column("is_saturated", coadd_cat["is_saturated"].data.astype(bool))
    coadd_cat.write(coadd_cat_file, overwrite=True)  # overwrite with additional info.


def recreate_coadd_cat(self):
    # NOTE: just in caes you need to recreate coadd with all information.
    sdss_data = self.get_sdss_data()
    wcs = sdss_data["wcs"]
    pixel_scale = sdss_data["pixel_scale"]
    add_extra_coadd_info(self.files["coadd_cat"], self.files["psf_image"], pixel_scale, wcs)


class BlissFigures:
    def __init__(self, outdir, cache="temp.pt", overwrite=False) -> None:

        self.outdir = Path(outdir)
        self.cache = self.outdir / cache
        self.overwrite = overwrite
        self.figures = {}

    @property
    @abstractmethod
    def fignames(self):
        """What figures will be produced with this class? What are their names?"""
        return {}

    def get_data(self, *args, **kwargs):
        """Return summary of data for producing plot, must be cachable w/ torch.save()."""
        if self.cache.exists() and not self.overwrite:
            return torch.load(self.cache)

        data = self.compute_data(*args, **kwargs)
        torch.save(data, self.cache)
        return data

    @abstractmethod
    def compute_data(self, *args, **kwargs):
        return {}

    def save_figures(self, *args, **kwargs):
        """Create figures and save to output directory with names from `self.fignames`."""
        data = self.get_data(*args, **kwargs)
        figs = self.create_figures(data)
        for k, fname in self.fignames.items():
            figs[k].savefig(self.outdir / fname, format="png")

    @abstractmethod
    def create_figures(self, data):
        """Return matplotlib figure instances to save based on data."""
        return mpl.figure.Figure()


class DetectionClassificationFigures(BlissFigures):
    def __init__(self, outdir="", cache="detect_class.pt", overwrite=False) -> None:
        super().__init__(outdir=outdir, cache=cache, overwrite=overwrite)

    @property
    def fignames(self):
        return {
            "detection": "sdss-precision-recall.png",
            "classification": "sdss-classification-acc.png",
        }

    def compute_data(self, scene, background, coadd_cat, encoder, decoder):
        assert isinstance(scene, (torch.Tensor, np.ndarray))
        assert encoder.device == decoder.device
        device = encoder.device

        bp = encoder.border_padding
        slen = 300  # chunk side-length
        h, w = scene.shape[-2], scene.shape[-1]
        hlims = (bp, h - bp)
        wlims = (bp, w - bp)

        # load coadd catalog
        coadd_params = reporting.get_params_from_coadd(coadd_cat, wlims, hlims)

        # misclassified galaxies in PHOTO as galaxies (obtaind by eye)
        ids = [8647475119820964111, 8647475119820964100, 8647475119820964192]
        for my_id in ids:
            idx = np.where(coadd_params["objid"] == my_id)[0].item()
            coadd_params["galaxy_bools"][idx] = 0

        # predict using models on scene.
        scene_torch = torch.from_numpy(scene).reshape(1, 1, h, w)
        background_torch = torch.from_numpy(background).reshape(1, 1, h, w)

        _, est_params = reconstruct_scene_at_coordinates(
            encoder,
            decoder,
            scene_torch,
            background_torch,
            hlims,
            wlims,
            slen=slen,
            device=device,
        )
        est_params["fluxes"] = (
            est_params["galaxy_bools"] * est_params["galaxy_fluxes"]
            + est_params["star_bools"] * est_params["fluxes"]
        )
        est_params["mags"] = sdss.convert_flux_to_mag(est_params["fluxes"])

        mag_bins = np.arange(18, 23, 0.25)  # skip 23
        precisions = []
        recalls = []
        class_accs = []
        galaxy_accs = []
        star_accs = []
        for mag in mag_bins:
            res = reporting.scene_metrics(
                coadd_params, est_params, mag_cut=mag, slack=1.0, mag_slack=0.5
            )
            precisions.append(res["precision"].item())
            recalls.append(res["recall"].item())
            class_accs.append(res["class_acc"].item())

            # how many out of the matched galaxies are accurately classified?
            galaxy_acc = res["conf_matrix"][0, 0] / res["conf_matrix"][0, :].sum()
            galaxy_accs.append(galaxy_acc)

            # how many out of the matched stars are correctly classified?
            star_acc = res["conf_matrix"][1, 1] / res["conf_matrix"][1, :].sum()
            star_accs.append(star_acc)

        return {
            "mag_bins": mag_bins,
            "precisions": precisions,
            "recalls": recalls,
            "class_accs": class_accs,
            "star_accs": star_accs,
            "galaxy_accs": galaxy_accs,
        }

    def create_figures(self, data):
        """Make figures related to detection and classification in SDSS."""
        sns.set_theme(style="darkgrid")

        mag_bins = data["mag_bins"]
        recalls = data["recalls"]
        precisions = data["precisions"]
        class_accs = data["class_accs"]
        galaxy_accs = data["galaxy_accs"]
        star_accs = data["star_accs"]

        reporting.set_rc_params(tick_label_size=22, label_size=30)
        f1, ax = plt.subplots(1, 1, figsize=(10, 10))
        reporting.format_plot(ax, xlabel=r"\rm magnitude cut", ylabel="value of metric")
        ax.plot(mag_bins, recalls, "-o", label=r"\rm recall")
        ax.plot(mag_bins, precisions, "-o", label=r"\rm precision")
        plt.xlim(18, 23)
        ax.legend(loc="best", prop={"size": 22})

        reporting.set_rc_params(tick_label_size=22, label_size=30)
        f2, ax = plt.subplots(1, 1, figsize=(10, 10))
        reporting.format_plot(ax, xlabel=r"\rm magnitude cut", ylabel="accuracy")
        ax.plot(mag_bins, class_accs, "-o", label=r"\rm classification accuracy")
        ax.plot(mag_bins, galaxy_accs, "-o", label=r"\rm galaxy classification accuracy")
        ax.plot(mag_bins, star_accs, "-o", label=r"\rm star classification accuracy")
        plt.xlim(18, 23)
        ax.legend(loc="best", prop={"size": 22})

        return {"detection": f1, "classification": f2}

    def scatter_plot_misclass(self, ax, galaxy_probs, misclass, true_mags):
        # TODO: Revive if useful later on.

        # scatter plot of miscclassification probs
        probs_correct = galaxy_probs[~misclass]
        probs_misclass = galaxy_probs[misclass]

        ax.scatter(true_mags[~misclass], probs_correct, marker="x", c="b")
        ax.scatter(true_mags[misclass], probs_misclass, marker="x", c="r")
        ax.axhline(0.5, linestyle="--")
        ax.axhline(0.1, linestyle="--")
        ax.axhline(0.9, linestyle="--")

        uncertain = (galaxy_probs[misclass] > 0.2) & (galaxy_probs[misclass] < 0.8)
        r_uncertain = sum(uncertain) / len(galaxy_probs[misclass])
        print(
            f"ratio misclass with probability between 10%-90%: {r_uncertain:.3f}",
        )


class SDSSReconstructionFigures(BlissFigures):
    def __init__(self, outdir="", cache="recon_sdss.pt", overwrite=False) -> None:
        super().__init__(outdir=outdir, cache=cache, overwrite=overwrite)

    @property
    def lims(self):
        """Specificy spatial limits on frame to obtain chunks to reconstruct."""

        # NOTE: Decoder assumes square images.
        return {
            "sdss_recon0": ((1700, 2000), (200, 500)),  # scene
            "sdss_recon1": ((1000, 1300), (1150, 1450)),  # scene
            "sdss_recon2": ((742, 790), (460, 508)),  # individual blend (both galaxies)
            "sdss_recon3": ((1710, 1758), (1400, 1448)),  # individual blend (both galaxies)
        }

    @property
    def fignames(self):
        return {**{f"sdss_recon{i}": f"sdss_reconstruction{i}.png" for i in range(len(self.lims))}}

    def compute_data(self, scene, background, coadd_cat, encoder, decoder):
        assert isinstance(scene, (torch.Tensor, np.ndarray))
        if not isinstance(scene, torch.Tensor):
            scene = torch.from_numpy(scene)
            background = torch.from_numpy(background)

        scene = scene.unsqueeze(0).unsqueeze(0)
        background = background.unsqueeze(0).unsqueeze(0)
        device = encoder.device

        bp = encoder.border_padding
        data = {}

        for figname in self.fignames:
            xlim, ylim = self.lims[figname]
            height, width = ylim[1] - ylim[0], xlim[1] - xlim[0]
            slen = min(height, width)
            assert height >= bp and width >= bp
            assert xlim[0] >= bp
            assert ylim[0] >= bp

            coadd_data = reporting.get_params_from_coadd(coadd_cat, xlim, ylim)
            with torch.no_grad():
                recon_image, recon_map = reconstruct_scene_at_coordinates(
                    encoder, decoder, scene, background, ylim, xlim, slen=slen, device=device
                )
            # only keep section inside border padding
            true_image = scene[0, 0, ylim[0] : ylim[1], xlim[0] : xlim[1]].cpu()
            recon_image = recon_image[0,0].cpu()
            res_image = (true_image - recon_image) / np.sqrt(recon_image)

            data[figname] = (true_image, recon_image, res_image, recon_map, coadd_data)

        return data

    def create_figures(self, data):  # pylint: disable=too-many-statements
        """Make figures related to reconstruction in SDSS."""
        out_figures = {}

        plt.style.use("seaborn-colorblind")
        pad = 6.0
        reporting.set_rc_params(fontsize=22, tick_label_size="small", legend_fontsize="small")
        for figname in self.fignames:
            true, recon, res, recon_map, coadd_data = data[figname]
            xlim, _ = self.lims[figname]
            scene_size = xlim[1] - xlim[0]
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

            locs_true = coadd_data["plocs"]
            true_galaxy_bools = coadd_data["galaxy_bools"]
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

            if recon_map is not None:
                locs_pred = recon_map["plocs"][0]
                star_bools = recon_map["star_bools"][0]
                galaxy_bools = recon_map["galaxy_bools"][0]
                locs_galaxies = locs_pred[galaxy_bools[:, 0] > 0.5, :]
                locs_stars = locs_pred[star_bools[:, 0] > 0.5, :]
                if locs_galaxies.shape[0] > 0:
                    in_bounds = torch.all(
                        (locs_galaxies > 0) & (locs_galaxies < scene_size), dim=-1
                    )
                    locs_galaxies = locs_galaxies[in_bounds]
                    ax_true.scatter(
                        locs_galaxies[:, 1], locs_galaxies[:, 0], color="c", marker="x", s=20
                    )
                    ax_recon.scatter(
                        locs_galaxies[:, 1],
                        locs_galaxies[:, 0],
                        color="c",
                        marker="x",
                        s=20,
                        label="Predicted Galaxy",
                        alpha=0.6,
                    )
                    ax_res.scatter(
                        locs_galaxies[:, 1], locs_galaxies[:, 0], color="c", marker="x", s=20
                    )
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

            out_figures[figname] = fig

        return out_figures


class AEReconstructionFigures(BlissFigures):
    def __init__(self, outdir="", cache="ae_cache.pt", overwrite=False, n_examples=5) -> None:
        super().__init__(outdir=outdir, cache=cache, overwrite=overwrite)
        self.n_examples = n_examples

    @property
    def fignames(self):
        return {
            "random_recon": "random_reconstructions.png",
            "worst_recon": "worst_reconstructions.png",
            "measure_contours": "single_galaxy_measurements_contours.png",
            "measure_scatter_bins": "single_galaxy_scatter_bins.png",
        }

    def compute_data(
        self, autoencoder: OneCenteredGalaxyAE, images_file, psf_file, sdss_pixel_scale
    ):
        # NOTE: For very dim objects (max_pixel < 6, flux ~ 1000), autoencoder can return
        # 0.1% objects with negative flux. This objects are discarded.
        device = autoencoder.device  # GPU is better otherwise slow.

        image_data = torch.load(images_file)
        images = image_data["images"]
        recon_means = torch.tensor([])
        background = image_data["background"].reshape(1, 1, 53, 53).to(device)
        noiseless_images = image_data["noiseless"].numpy()  # no background or noise.

        print("Computing reconstructions from saved autoencoder model...")
        n_images = images.shape[0]
        batch_size = 128
        n_iters = int(np.ceil(n_images // 128))
        for i in range(n_iters):  # in batches otherwise GPU error.
            bimages = images[batch_size * i : batch_size * (i + 1)].to(device)
            recon_mean, _ = autoencoder.forward(bimages, background)
            recon_mean = recon_mean.detach().to("cpu")
            recon_means = torch.cat((recon_means, recon_mean))
        residuals = (images - recon_means) / recon_means.sqrt()
        assert recon_means.shape[0] == noiseless_images.shape[0]

        # random
        rand_indices = torch.randint(0, len(images), size=(self.n_examples,))

        # worst
        absolute_residual = residuals.abs().sum(axis=(1, 2, 3))
        worst_indices = absolute_residual.argsort()[-self.n_examples :]

        # measurements
        psf_array = np.load(psf_file)
        galsim_psf_image = galsim.Image(psf_array[0], scale=sdss_pixel_scale)  # sdss scale
        psf = galsim.InterpolatedImage(galsim_psf_image).withFlux(1.0)
        psf_image = psf.drawImage(nx=53, ny=53, scale=sdss_pixel_scale).array

        recon_no_background = recon_means.numpy() - background.cpu().numpy()
        # a small percentage of low magnitude objects end up predicted with negative flux.
        good_bool = recon_no_background.sum(axis=(1, 2, 3)) > 0
        measurements = reporting.get_single_galaxy_measurements(
            slen=53,
            true_images=noiseless_images[good_bool],
            recon_images=recon_no_background[good_bool],
            psf_image=psf_image.reshape(-1, 53, 53),
            pixel_scale=sdss_pixel_scale,
        )

        return {
            "random": (images[rand_indices], recon_means[rand_indices], residuals[rand_indices]),
            "worst": (images[worst_indices], recon_means[worst_indices], residuals[worst_indices]),
            "measurements": measurements,
        }

    def reconstruction_figure(self, images, recons, residuals):
        plt.style.use("seaborn-colorblind")
        pad = 6.0
        reporting.set_rc_params(fontsize=22, tick_label_size="small", legend_fontsize="small")
        fig, axes = plt.subplots(nrows=self.n_examples, ncols=3, figsize=(12, 20))

        assert images.shape[0] == recons.shape[0] == residuals.shape[0] == self.n_examples
        assert images.shape[1] == recons.shape[1] == residuals.shape[1] == 1, "1 band only."

        # pick standard ranges for residuals
        vmin_res = residuals.min().item()
        vmax_res = residuals.max().item()

        for i in range(self.n_examples):

            ax_true = axes[i, 0]
            ax_recon = axes[i, 1]
            ax_res = axes[i, 2]

            # only add titles to the first axes.
            if i == 0:
                ax_true.set_title("Images $x$", pad=pad)
                ax_recon.set_title(r"Reconstruction $\tilde{x}$", pad=pad)
                ax_res.set_title(
                    r"Residual $\left(x - \tilde{x}\right) / \sqrt{\tilde{x}}$", pad=pad
                )

            # standarize ranges of true and reconstruction
            image = images[i, 0].detach().cpu().numpy()
            recon = recons[i, 0].detach().cpu().numpy()
            residual = residuals[i, 0].detach().cpu().numpy()

            vmin = min(image.min().item(), recon.min().item())
            vmax = max(image.max().item(), recon.max().item())

            # plot images
            reporting.plot_image(fig, ax_true, image, vrange=(vmin, vmax))
            reporting.plot_image(fig, ax_recon, recon, vrange=(vmin, vmax))
            reporting.plot_image(fig, ax_res, residual, vrange=(vmin_res, vmax_res))

        plt.subplots_adjust(hspace=-0.4)
        plt.tight_layout()

        return fig

    def make_scatter_contours(self, ax, x, y, **plot_kwargs):
        sns.scatterplot(x=x, y=y, s=10, color="0.15", ax=ax)
        sns.histplot(x=x, y=y, pthresh=0.1, cmap="mako", ax=ax, cbar=True)
        sns.kdeplot(x=x, y=y, levels=10, color="w", linewidths=1, ax=ax)
        reporting.format_plot(ax, **plot_kwargs)

    def make_2d_hist(self, x, y, color="m", height=7, **plot_kwargs):
        # NOTE: This creates its own figure object which makes it hard to use.
        # TODO: Revive if useful later on.
        g = sns.jointplot(x=x, y=y, color=color, kind="hist", marginal_ticks=True, height=height)
        g.ax_joint.axline(xy1=(np.median(x), np.median(y)), slope=1.0)
        reporting.format_plot(g.ax_joint, **plot_kwargs)

    def scatter_bin_plot(self, ax, x, y, xlims, delta, capsize=5.0, **plot_kwargs):

        xbins = np.arange(xlims[0], xlims[1], delta)

        xs = np.zeros(len(xbins))
        ys = np.zeros(len(xbins))
        errs = np.zeros((len(xbins), 2))

        for i, bx in enumerate(xbins):
            keep_x = (x > bx) & (x < bx + delta)
            y_bin = y[keep_x]

            xs[i] = bx + delta / 2

            if len(y_bin) == 0:  # noqa: WPS507
                ys[i] = np.nan
                errs[i] = (np.nan, np.nan)
                continue

            ys[i] = np.median(y_bin)
            errs[i, :] = ys[i] - np.quantile(y_bin, 0.25), np.quantile(y_bin, 0.75) - ys[i]

        errs = errs.T.reshape(2, -1)
        ax.errorbar(xs, ys, yerr=errs, marker="o", c="m", linestyle="--", capsize=capsize)
        reporting.format_plot(ax, **plot_kwargs)

    def make_scatter_contours_plot(self, meas):
        sns.set_theme(style="darkgrid")
        reporting.set_rc_params(
            fontsize=22, legend_fontsize="small", tick_label_size="small", label_size="medium"
        )
        fig, axes = plt.subplots(2, 2, figsize=(12, 12))
        ax1, ax2, ax3, ax4 = axes.flatten()

        # fluxes / magnitudes
        x, y = remove_outliers(meas["true_mags"], meas["recon_mags"], level=0.95)
        mag_ticks = (16, 17, 18, 19)
        xlabel = r"\rm true mag."
        ylabel = r"\rm recon mag."
        self.make_scatter_contours(
            ax1, x, y, xlabel=xlabel, ylabel=ylabel, xticks=mag_ticks, yticks=mag_ticks
        )

        # hlrs
        x, y = remove_outliers(meas["true_hlrs"], meas["recon_hlrs"], level=0.95)
        self.make_scatter_contours(ax2, x, y, xlabel=r"$r^{\rm true}$", ylabel=r"$r^{\rm recon}$")

        # ellipticities 1
        x, y = remove_outliers(meas["true_ellip"][:, 0], meas["recon_ellip"][:, 0], level=0.95)
        g_ticks = (-1.0, -0.5, 0.0, 0.5, 1.0)
        xlabel = r"$g_{1}^{\rm true}$"
        ylabel = r"$g_{1}^{\rm recon}$"
        self.make_scatter_contours(
            ax3, x, y, xticks=g_ticks, yticks=g_ticks, xlabel=xlabel, ylabel=ylabel
        )

        # ellipticities 2
        x, y = remove_outliers(meas["true_ellip"][:, 1], meas["recon_ellip"][:, 1], level=0.95)
        xlabel = r"$g_{2}^{\rm true}$"
        ylabel = r"$g_{2}^{\rm recon}$"
        self.make_scatter_contours(
            ax4, x, y, xticks=g_ticks, yticks=g_ticks, xlabel=xlabel, ylabel=ylabel
        )

        plt.tight_layout()
        return fig

    def make_scatter_bin_plots(self, meas):
        fig, axes = plt.subplots(2, 2, figsize=(16, 16))
        ax1, ax2, ax3, ax4 = axes.flatten()
        reporting.set_rc_params(fontsize=24)

        # fluxes / magnitudes
        true_mags, recon_mags = meas["true_mags"], meas["recon_mags"]
        x, y = remove_outliers(true_mags, (recon_mags - true_mags) / true_mags, level=0.99)
        self.scatter_bin_plot(
            ax1,
            x,
            y,
            xlims=(x.min(), x.max()),
            delta=0.25,
            xlabel=r"\rm true mag.",
            ylabel=r"\rm mag. relative error",
            xticks=[16, 17, 18, 19, 20],
        )

        # hlrs
        true_hlrs, recon_hlrs = meas["true_hlrs"], meas["recon_hlrs"]
        x, y = remove_outliers(true_hlrs, (recon_hlrs - true_hlrs) / true_hlrs, level=0.99)
        self.scatter_bin_plot(
            ax2,
            x,
            y,
            xlims=(x.min(), x.max()),
            delta=0.5,
            xlabel=r"$r^{\rm true}$",
            ylabel=r"$(r^{\rm recon} - r^{\rm true}) / r^{\rm true}$",
        )

        # ellipticities
        true_ellip1, recon_ellip1 = meas["true_ellip"][:, 0], meas["recon_ellip"][:, 0]
        x, y = remove_outliers(true_ellip1, recon_ellip1 - true_ellip1, level=0.99)
        self.scatter_bin_plot(
            ax3,
            x,
            y,
            xlims=(-0.85, 0.85),
            delta=0.1,
            xticks=[-1.0, -0.5, 0.0, 0.5, 1.0],
            xlabel=r"$g_{1}^{\rm true}$",
            ylabel=r"$g_{1}^{\rm recon} - g_{1}^{\rm true}$",
        )

        true_ellip2, recon_ellip2 = meas["true_ellip"][:, 1], meas["recon_ellip"][:, 1]
        x, y = remove_outliers(true_ellip2, recon_ellip2 - true_ellip2, level=0.99)
        self.scatter_bin_plot(
            ax4,
            x,
            y,
            xlims=(-0.85, 0.85),
            delta=0.25,
            xticks=[-1.0, -0.5, 0.0, 0.5, 1.0],
            xlabel=r"$g_{2}^{\rm true}$",
            ylabel=r"$g_{2}^{\rm recon} - g_{2}^{\rm true}$",
        )

        plt.tight_layout()

        return fig

    def create_figures(self, data):

        return {
            "random_recon": self.reconstruction_figure(*data["random"]),
            "worst_recon": self.reconstruction_figure(*data["worst"]),
            "measure_contours": self.make_scatter_contours_plot(data["measurements"]),
            "measure_scatter_bins": self.make_scatter_bin_plots(data["measurements"]),
        }


@hydra.main(config_path="./config", config_name="config")
def plots(cfg):

    fig = set(cfg.plots.fig)
    outdir = cfg.plots.outdir
    overwrite = cfg.plots.overwrite
    device = torch.device(cfg.plots.device)

    if not Path(outdir).exists():
        warnings.warn("Specified output directory does not exist, will attempt to create it.")
        Path(outdir).mkdir(exist_ok=True, parents=True)

    # load models required for SDSS reconstructions.
    if fig.intersection({2, 3}):
        sleep = instantiate(cfg.models.sleep)
        sleep.load_state_dict(torch.load(cfg.predict.sleep_checkpoint))
        location = sleep.image_encoder.to(device).eval()

        binary = instantiate(cfg.models.binary)
        binary.load_state_dict(torch.load(cfg.predict.binary_checkpoint))
        binary = binary.to(device).eval()

        galaxy = instantiate(cfg.models.galaxy_encoder)
        galaxy.load_state_dict(torch.load(cfg.predict.galaxy_checkpoint))
        galaxy = galaxy.to(device).eval()

        decoder = sleep.image_decoder.to(device).eval()
        encoder = Encoder(location.eval(), binary.eval(), galaxy.eval()).to(device)

    # FIGURE 1: Autoencoder single galaxy reconstruction
    if 1 in fig:
        autoencoder = instantiate(cfg.models.galaxy_net)
        autoencoder.load_state_dict(torch.load(cfg.models.prior.galaxy_prior.autoencoder_ckpt))
        autoencoder = autoencoder.to(device).eval()

        # generate galsim simulated galaxies images if file does not exist.
        galaxies_file = Path(cfg.plots.simulated_sdss_individual_galaxies)
        if not galaxies_file.exists():
            print(f"Generating individual galaxy images and saving to: {galaxies_file}")
            dataset = instantiate(
                cfg.datasets.sdss_galaxies, batch_size=512, n_batches=20, num_workers=20
            )
            imagepath = galaxies_file.parent / (galaxies_file.stem + "_images.pdf")
            generate.generate(
                dataset, galaxies_file, imagepath, n_plots=25, global_params=("background", "slen")
            )

        # create figure classes and plot.
        ae_figures = AEReconstructionFigures(outdir, overwrite=overwrite, n_examples=5)
        ae_figures.save_figures(
            autoencoder, galaxies_file, cfg.plots.psf_file, cfg.plots.sdss_pixel_scale
        )
        mpl.rc_file_defaults()

    # FIGURE 2: Classification and Detection metrics
    if 2 in fig:
        scene = get_sdss_data(cfg)["image"]
        background = get_sdss_data(cfg)["background"]
        coadd_cat = Table.read(cfg.plots.coadd_cat, format="fits")
        dc_fig = DetectionClassificationFigures(outdir, overwrite=overwrite)
        dc_fig.save_figures(scene, background, coadd_cat, encoder, decoder)
        mpl.rc_file_defaults()

    # FIGURE 3: Reconstructions on SDSS
    if 3 in fig:
        scene = get_sdss_data(cfg)["image"]
        background = get_sdss_data(cfg)["background"]
        sdss_rec_fig = SDSSReconstructionFigures(outdir, overwrite=overwrite)
        sdss_rec_fig.save_figures(scene, background, coadd_cat, encoder, decoder)
        mpl.rc_file_defaults()

    else:
        raise NotImplementedError("The figure specified has not been created.")


if __name__ == "__main__":
    plots()  # pylint: disable=no-value-for-parameter
