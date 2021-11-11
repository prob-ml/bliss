#!/usr/bin/env python3
"""Produce all figures. Save to nice PDF format."""
import argparse
import os
from abc import abstractmethod
from pathlib import Path

import galsim
import matplotlib as mpl
import numpy as np
import pytorch_lightning as pl
import seaborn as sns
import torch
from astropy.table import Table
from astropy.wcs.wcs import WCS
from hydra import compose, initialize
from matplotlib import pyplot as plt

from bliss import generate, reporting
from bliss.datasets import sdss
from bliss.datasets.galsim_galaxies import load_psf_from_file
from bliss.models.binary import BinaryEncoder
from bliss.models.galaxy_encoder import GalaxyEncoder
from bliss.models.galaxy_net import OneCenteredGalaxyAE
from bliss.predict import predict_on_scene
from bliss.sleep import SleepPhase

device = torch.device("cuda:0")
pl.seed_everything(0)

files_dict = {
    "sleep_ckpt": "models/sdss_sleep.ckpt",
    "galaxy_encoder_ckpt": "models/sdss_galaxy_encoder.ckpt",
    "binary_ckpt": "models/sdss_binary.ckpt",
    "ae_ckpt": "models/sdss_autoencoder.ckpt",
    "coadd_cat": "data/coadd_catalog_94_1_12.fits",
    "sdss_dir": "data/sdss",
    "psf_file": "data/psField-000094-1-0012-PSF-image.npy",
}

SDSS_PIXEL_SCALE = 0.396


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


def add_extra_coadd_info(coadd_cat_file: str, psf_image_file: str, pixel_scale: float, wcs: WCS):
    """Add additional useful information to coadd catalog."""
    coadd_cat = Table.read(coadd_cat_file)

    psf = load_psf_from_file(psf_image_file, pixel_scale)
    x, y = wcs.all_world2pix(coadd_cat["ra"], coadd_cat["dec"], 0)
    galaxy_bool = ~coadd_cat["probpsf"].data.astype(bool)
    flux, mag = reporting.get_flux_coadd(coadd_cat)
    hlr = reporting.get_hlr_coadd(coadd_cat, psf)

    coadd_cat["x"] = x
    coadd_cat["y"] = y
    coadd_cat["galaxy_bool"] = galaxy_bool
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
    def __init__(self, outdir="", cache="temp.pt", overwrite=False) -> None:
        os.chdir(os.getenv("BLISS_HOME"))
        outdir = Path(outdir)

        if not outdir.exists():
            outdir.mkdir(exist_ok=True)

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
            figs[k].savefig(self.outdir / fname, format="pdf")

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
            "detection": "sdss-precision-recall.pdf",
            "classification": "sdss-classification-acc.pdf",
        }

    def compute_data(self, scene, coadd_cat, sleep_net, binary_encoder, galaxy_encoder):
        assert isinstance(scene, (torch.Tensor, np.ndarray))
        assert sleep_net.device == binary_encoder.device == galaxy_encoder.device
        device = sleep_net.device

        bp = 24
        clen = 300
        h, w = scene.shape[-2], scene.shape[-1]

        # load coadd catalog
        coadd_params = reporting.get_params_from_coadd(coadd_cat, h, w, bp)

        # misclassified galaxies in PHOTO as galaxies (obtaind by eye)
        ids = [8647475119820964111, 8647475119820964100, 8647475119820964192]
        for my_id in ids:
            idx = np.where(coadd_params["objid"] == my_id)[0].item()
            coadd_params["galaxy_bool"][idx] = 0

        # load specific models that are needed.
        image_encoder = sleep_net.image_encoder.to(device).eval()
        galaxy_decoder = sleep_net.image_decoder.galaxy_tile_decoder.galaxy_decoder.eval()

        # predict using models on scene.
        scene_torch = torch.from_numpy(scene).reshape(1, 1, h, w)
        _, est_params = predict_on_scene(
            clen,
            scene_torch,
            image_encoder,
            binary_encoder,
            galaxy_encoder,
            galaxy_decoder,
            device,
        )

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

    def scatter_plot_misclass(self, ax, prob_galaxy, misclass, true_mags):
        # TODO: Revive if necessary later on.

        # scatter plot of miscclassification probs
        probs_correct = prob_galaxy[~misclass]
        probs_misclass = prob_galaxy[misclass]

        ax.scatter(true_mags[~misclass], probs_correct, marker="x", c="b")
        ax.scatter(true_mags[misclass], probs_misclass, marker="x", c="r")
        ax.axhline(0.5, linestyle="--")
        ax.axhline(0.1, linestyle="--")
        ax.axhline(0.9, linestyle="--")

        uncertain = (prob_galaxy[misclass] > 0.2) & (prob_galaxy[misclass] < 0.8)
        r_uncertain = sum(uncertain) / len(prob_galaxy[misclass])
        print(
            f"ratio misclass with probability between 10%-90%: {r_uncertain:.3f}",
        )


class AEReconstructionFigures(BlissFigures):
    def __init__(self, outdir="", cache="ae_cache.pt", overwrite=False, n_examples=5) -> None:
        super().__init__(outdir=outdir, cache=cache, overwrite=overwrite)
        self.n_examples = n_examples

    @property
    def fignames(self):
        return {
            "random_recon": "random_reconstructions.pdf",
            "worst_recon": "worst_reconstructions.pdf",
            "measure_contours": "single_galaxy_measurements_contours.pdf",
            "measure_scatter_bins": "single_galaxy_scatter_bins.pdf",
        }

    def compute_data(self, autoencoder, images_file, psf_file):
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
            recon_mean = autoencoder.forward(bimages, background).detach().to("cpu")
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
        galsim_psf_image = galsim.Image(psf_array[0], scale=SDSS_PIXEL_SCALE)  # sdss scale
        psf = galsim.InterpolatedImage(galsim_psf_image).withFlux(1.0)
        psf_image = psf.drawImage(nx=53, ny=53, scale=SDSS_PIXEL_SCALE).array

        recon_no_background = recon_means.numpy() - background.cpu().numpy()
        # a small percentage of low magnitude objects end up predicted with negative flux.
        good_bool = recon_no_background.sum(axis=(1, 2, 3)) > 0
        measurements = reporting.get_single_galaxy_measurements(
            slen=53,
            true_images=noiseless_images[good_bool],
            recon_images=recon_no_background[good_bool],
            psf_image=psf_image.reshape(-1, 53, 53),
            pixel_scale=SDSS_PIXEL_SCALE,
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


def main(n_fig, overwrite=False):
    os.chdir(os.getenv("BLISS_HOME"))  # simplicity for I/O
    outdir = "output/sdss_figures"

    if n_fig == 1:
        # FIGURE 1: Autoencoder performance.
        # first, create images of individually simulated galaxies if they do not exist.
        autoencoder = OneCenteredGalaxyAE.load_from_checkpoint(files_dict["ae_ckpt"]).eval()
        autoencoder = autoencoder.eval().to(device)
        galaxies_file = Path("data/simulated_sdss_individual_galaxies.pt")
        if not galaxies_file.exists():
            print(f"Generating individual galaxy images and saving to: {galaxies_file}")
            overrides = ["experiment=sdss_individual_galaxies"]
            with initialize(config_path="config"):
                cfg = compose("config", overrides=overrides)
                generate.generate(cfg)

        # create figure classes and plot.
        ae_figures = AEReconstructionFigures(outdir=outdir, overwrite=overwrite, n_examples=5)
        ae_figures.save_figures(autoencoder, galaxies_file, files_dict["psf_file"])

    # FIGURE 2: Classification and Detection metrics
    elif n_fig == 2:
        scene = get_sdss_data()["image"]
        coadd_cat = Table.read(files_dict["coadd_cat"], format="fits")
        sleep_net = SleepPhase.load_from_checkpoint(files_dict["sleep_ckpt"]).to(device)
        binary_encoder = BinaryEncoder.load_from_checkpoint(files_dict["binary_ckpt"])
        binary_encoder = binary_encoder.to(device).eval()
        galaxy_encoder = GalaxyEncoder.load_from_checkpoint(files_dict["galaxy_encoder_ckpt"])
        galaxy_encoder = galaxy_encoder.to(device).eval()

        dc_fig = DetectionClassificationFigures(outdir=outdir, overwrite=overwrite)
        dc_fig.save_figures(scene, coadd_cat, sleep_net, binary_encoder, galaxy_encoder)

    else:
        raise NotImplementedError("That Figure has not been created.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create figures related to SDSS galaxies.")
    parser.add_argument(
        "-f",
        "--fig",
        help="Which figures do you want to create?",
        required=True,
        choices=["1", "2"],
    )
    parser.add_argument(
        "-o", "--overwrite", action="store_true", default=False, help="Recreate cache?"
    )
    args = vars(parser.parse_args())
    n_fig = int(args["fig"])
    main(n_fig, args["overwrite"])
