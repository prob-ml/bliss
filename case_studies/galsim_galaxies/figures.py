#!/usr/bin/env python3
import warnings
from pathlib import Path

import hydra
import matplotlib as mpl
import numpy as np
import torch
from hydra.utils import instantiate
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from torch import Tensor
from tqdm import tqdm

from bliss import generate, reporting
from bliss.catalog import TileCatalog
from bliss.encoder import Encoder
from bliss.models.decoder import ImageDecoder
from bliss.models.galaxy_net import OneCenteredGalaxyAE
from bliss.models.psf_decoder import PSFDecoder
from bliss.plotting import BlissFigure, plot_image, scatter_bin_plot
from bliss.reporting import match_by_locs

ALL_FIGS = {"single_gal", "blend_gal"}


class AutoEncoderReconRandom(BlissFigure):
    def __init__(self, *args, n_examples: int = 5, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.n_examples = n_examples

    @property
    def rc_kwargs(self):
        return {"fontsize": 22, "tick_label_size": "small", "legend_fontsize": "small"}

    @property
    def cache_name(self) -> str:
        return "ae"

    @property
    def name(self) -> str:
        return "ae_recon_random"

    def compute_data(
        self,
        autoencoder: OneCenteredGalaxyAE,
        images_file: str,
        psf_params_file: str,
        sdss_pixel_scale: float,
    ):
        device = autoencoder.device  # GPU is better otherwise slow.
        image_data = torch.load(images_file)
        true_params: Tensor = image_data["params"]
        images: Tensor = image_data["images"]
        recon_means = torch.tensor([])
        background: Tensor = image_data["background"].reshape(1, 1, 53, 53).to(device)
        noiseless_images: Tensor = image_data["noiseless"]
        snr: Tensor = image_data["snr"].reshape(-1)

        print("INFO: Computing reconstructions from saved autoencoder model...")
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
        psf_decoder = PSFDecoder(psf_params_file=psf_params_file, psf_slen=53, sdss_bands=[2])
        psf_image = psf_decoder.forward_psf_from_params()

        recon_no_background = recon_means - background.cpu()
        assert torch.all(recon_no_background.sum(axis=(1, 2, 3)) > 0)
        true_meas = reporting.get_single_galaxy_measurements(
            noiseless_images, psf_image.reshape(-1, 53, 53), sdss_pixel_scale
        )
        true_meas = {f"true_{k}": v for k, v in true_meas.items()}
        recon_meas = reporting.get_single_galaxy_measurements(
            recon_no_background, psf_image.reshape(-1, 53, 53), sdss_pixel_scale
        )
        recon_meas = {f"recon_{k}": v for k, v in recon_meas.items()}
        measurements = {**true_meas, **recon_meas, "snr": snr}

        return {
            "random": {
                "true": images[rand_indices],
                "recon": recon_means[rand_indices],
                "res": residuals[rand_indices],
            },
            "worst": {
                "true": images[worst_indices],
                "recon": recon_means[worst_indices],
                "res": residuals[worst_indices],
            },
            "measurements": measurements,
            "true_params": true_params,
        }

    def _reconstruction_figure(self, images, recons, residuals) -> Figure:

        pad = 6.0
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
            image = images[i, 0]
            recon = recons[i, 0]
            residual = residuals[i, 0]

            vmin = min(image.min().item(), recon.min().item())
            vmax = max(image.max().item(), recon.max().item())

            # plot images
            plot_image(fig, ax_true, image, vrange=(vmin, vmax))
            plot_image(fig, ax_recon, recon, vrange=(vmin, vmax))
            plot_image(fig, ax_res, residual, vrange=(vmin_res, vmax_res))

        plt.subplots_adjust(hspace=-0.4)
        plt.tight_layout()

        return fig

    def create_figure(self, data) -> Figure:
        return self._reconstruction_figure(*data["random"].values())


class AutoEncoderBinMeasurements(AutoEncoderReconRandom):
    @property
    def name(self) -> str:
        return "ae_bin_residuals"

    @property
    def rc_kwargs(self):
        return {"fontsize": 24}

    def create_figure(self, data) -> Figure:
        meas = data["measurements"]

        fig, axes = plt.subplots(2, 2, figsize=(16, 16))
        ((ax1, ax2), (ax3, ax4)) = axes
        snr = meas["snr"]
        xticks = [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
        xlims = (0, 3)
        xlabel = r"$\log_{10} \text{SNR}$"

        # magnitudes
        true_mags, recon_mags = meas["true_mags"], meas["recon_mags"]
        x, y = np.log10(snr), recon_mags - true_mags
        scatter_bin_plot(ax1, x, y, xlims, delta=0.2)
        ax1.set_xlim(xlims)
        ax1.set_xlabel(xlabel)
        ax1.set_ylabel(r"\rm $m^{\rm recon} - m^{\rm true}$")
        ax1.set_xticks(xticks)

        # fluxes
        true_fluxes, recon_fluxes = meas["true_fluxes"], meas["recon_fluxes"]
        x, y = np.log10(snr), (recon_fluxes - true_fluxes) / recon_fluxes
        scatter_bin_plot(ax2, x, y, xlims, delta=0.2)
        ax2.set_xlim(xlims)
        ax2.set_xlabel(xlabel)
        ax2.set_ylabel(r"\rm $(f^{\rm recon} - f^{\rm true}) / f^{\rm true}$")
        ax2.set_xticks(xticks)

        # ellipticities
        true_ellip1, recon_ellip1 = meas["true_ellips"][:, 0], meas["recon_ellips"][:, 0]
        x, y = np.log10(snr), recon_ellip1 - true_ellip1
        scatter_bin_plot(ax3, x, y, xlims, delta=0.2)
        ax3.set_xlim(xlims)
        ax3.set_xlabel(xlabel)
        ax3.set_ylabel(r"$g_{1}^{\rm recon} - g_{1}^{\rm true}$")
        ax3.set_xticks(xticks)

        true_ellip2, recon_ellip2 = meas["true_ellips"][:, 1], meas["recon_ellips"][:, 1]
        x, y = np.log10(snr), recon_ellip2 - true_ellip2
        scatter_bin_plot(ax4, x, y, xlims, delta=0.2)
        ax4.set_xlim(xlims)
        ax4.set_xlabel(xlabel)
        ax4.set_ylabel(r"$g_{2}^{\rm recon} - g_{2}^{\rm true}$")
        ax4.set_xticks(xticks)

        return fig


class BlendGalsimFigure(BlissFigure):
    @property
    def rc_kwargs(self):
        return {"fontsize": 24}

    @property
    def cache_name(self) -> str:
        return "blend_sim"

    @property
    def name(self) -> str:
        return "blendsim_gal_meas"

    def compute_data(self, blend_file: Path, encoder: Encoder, decoder: ImageDecoder):
        blend_data = torch.load(blend_file)
        images = blend_data.pop("images")
        background = blend_data.pop("background")
        n_batches, _, slen, _ = images.shape
        assert background.shape == (1, slen, slen)

        # prepare background
        background = background.unsqueeze(0)
        background = background.expand(n_batches, 1, slen, slen)

        # first create FullCatalog from simulated data
        tile_cat = TileCatalog(decoder.tile_slen, blend_data).cpu()
        full_truth = tile_cat.to_full_params()

        print("INFO: BLISS posterior inference on images.")
        tile_est = encoder.variational_mode(images, background)
        tile_est.set_all_fluxes_and_mags(decoder)
        tile_est.set_galaxy_ellips(decoder, scale=0.393)
        tile_est = tile_est.cpu()
        full_est = tile_est.to_full_params()

        snr = []
        blendedness = []
        true_mags = []
        true_ellips1 = []
        true_ellips2 = []
        est_mags = []
        est_ellips1 = []
        est_ellips2 = []
        for ii in tqdm(range(n_batches), desc="Matching batches"):
            true_plocs_ii, est_plocs_ii = full_truth.plocs[ii], full_est.plocs[ii]

            tindx, eindx, dkeep, _ = match_by_locs(true_plocs_ii, est_plocs_ii)
            n_matches = len(tindx[dkeep])

            snr_ii = full_truth["snr"][ii][tindx][dkeep]
            blendedness_ii = full_truth["blendedness"][ii][tindx][dkeep]
            true_mag_ii = full_truth["mags"][ii][tindx][dkeep]
            est_mag_ii = full_est["mags"][ii][eindx][dkeep]
            true_ellips_ii = full_truth["ellips"][ii][tindx][dkeep]
            est_ellips_ii = full_est["ellips"][ii][eindx][dkeep]
            n_matches = len(snr_ii)
            for jj in range(n_matches):
                snr.append(snr_ii[jj].item())
                blendedness.append(blendedness_ii[jj].item())
                true_mags.append(true_mag_ii[jj].item())
                est_mags.append(est_mag_ii[jj].item())
                true_ellips1.append(true_ellips_ii[jj][0].item())
                true_ellips2.append(true_ellips_ii[jj][1].item())
                est_ellips1.append(est_ellips_ii[jj][0].item())
                est_ellips2.append(est_ellips_ii[jj][1].item())

        true_ellips = torch.vstack([torch.tensor(true_ellips1), torch.tensor(true_ellips2)])
        true_ellips = true_ellips.T.reshape(-1, 2)

        est_ellips = torch.vstack([torch.tensor(est_ellips1), torch.tensor(est_ellips2)])
        est_ellips = est_ellips.T.reshape(-1, 2)

        return {
            "snr": torch.tensor(snr),
            "blendedness": torch.tensor(blendedness),
            "true_mags": torch.tensor(true_mags),
            "est_mags": torch.tensor(est_mags),
            "true_ellips": true_ellips,
            "est_ellips": est_ellips,
        }

    def create_figure(self, data):
        snr, blendedness, true_mags, est_mags, true_ellips, est_ellips = data.values()
        fig, axes = plt.subplots(3, 2, figsize=(12, 18))
        ax1, ax2, ax3, ax4, ax5, ax6 = axes.flatten()

        xticks = [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
        xlims = (0, 3)
        xlabel = r"$\log_{10} \rm SNR$"
        ylabel = r"\rm $m^{\rm recon} - m^{\rm true}$"

        x, y = np.log10(snr), est_mags - true_mags
        scatter_bin_plot(ax1, x, y, xlims, delta=0.2)
        ax1.set_xlabel(xlabel)
        ax1.set_ylabel(ylabel)
        ax1.set_xticks(xticks)

        xticks = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
        xlims = (0, 1)
        xlabel = "$B$"
        ylabel = r"\rm $m^{\rm recon} - m^{\rm true}$"
        x, y = blendedness, est_mags - true_mags
        scatter_bin_plot(ax2, x, y, xlims, delta=0.1)
        ax2.set_xlabel(xlabel)
        ax2.set_ylabel(ylabel)
        ax2.set_xticks(xticks)

        xticks = [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
        xlims = (0, 3)
        xlabel = r"$\log_{10} \rm SNR$"
        ylabel = r"$g_{1}^{\rm recon} - g_{1}^{\rm true}$"
        x, y = np.log10(snr), est_ellips[:, 0] - true_ellips[:, 0]
        scatter_bin_plot(ax3, x, y, xlims, delta=0.2)
        ax3.set_xlabel(xlabel)
        ax3.set_ylabel(ylabel)
        ax3.set_xticks(xticks)

        xticks = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
        xlims = (0, 1)
        xlabel = "$B$"
        ylabel = r"$g_{1}^{\rm recon} - g_{1}^{\rm true}$"
        x, y = blendedness, est_ellips[:, 0] - true_ellips[:, 0]
        scatter_bin_plot(ax4, x, y, xlims, delta=0.1)
        ax4.set_xlabel(xlabel)
        ax4.set_ylabel(ylabel)
        ax4.set_xticks(xticks)

        xticks = [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
        xlims = (0, 3)
        xlabel = r"$\log_{10} \rm SNR$"
        ylabel = r"$g_{2}^{\rm recon} - g_{2}^{\rm true}$"
        x, y = np.log10(snr), est_ellips[:, 1] - true_ellips[:, 1]
        scatter_bin_plot(ax5, x, y, xlims, delta=0.2)
        ax5.set_xlabel(xlabel)
        ax5.set_ylabel(ylabel)
        ax5.set_xticks(xticks)

        xticks = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
        xlims = (0, 1)
        xlabel = "$B$"
        ylabel = r"$g_{2}^{\rm recon} - g_{2}^{\rm true}$"
        x, y = blendedness, est_ellips[:, 1] - true_ellips[:, 1]
        scatter_bin_plot(ax6, x, y, xlims=xlims, delta=0.1)
        ax6.set_xlabel(xlabel)
        ax6.set_ylabel(ylabel)
        ax6.set_xticks(xticks)

        plt.tight_layout()

        return fig


def load_models(cfg, device):
    # load models required for SDSS reconstructions.

    location = instantiate(cfg.models.detection_encoder).to(device).eval()
    location.load_state_dict(
        torch.load(cfg.plots.location_checkpoint, map_location=location.device)
    )

    binary = instantiate(cfg.models.binary).to(device).eval()
    binary.load_state_dict(torch.load(cfg.plots.binary_checkpoint, map_location=binary.device))

    galaxy = instantiate(cfg.models.galaxy_encoder).to(device).eval()
    galaxy.load_state_dict(torch.load(cfg.plots.galaxy_checkpoint, map_location=galaxy.device))

    n_images_per_batch = cfg.plots.encoder.n_images_per_batch
    n_rows_per_batch = cfg.plots.encoder.n_rows_per_batch
    encoder = Encoder(
        location.eval(),
        binary.eval(),
        galaxy.eval(),
        n_images_per_batch=n_images_per_batch,
        n_rows_per_batch=n_rows_per_batch,
    )
    encoder = encoder.to(device)
    decoder: ImageDecoder = instantiate(cfg.models.decoder).to(device).eval()
    return encoder, decoder


def setup(cfg):
    pcfg = cfg.plots
    figs = set(pcfg.figs)
    cachedir = pcfg.cachedir
    device = torch.device(pcfg.device)
    bfig_kwargs = {
        "figdir": pcfg.figdir,
        "cachedir": cachedir,
        "img_format": pcfg.image_format,
    }

    if not Path(cachedir).exists():
        warnings.warn("Specified cache directory does not exist, will attempt to create it.")
        Path(cachedir).mkdir(exist_ok=True, parents=True)

    assert set(figs).issubset(ALL_FIGS)
    return figs, device, bfig_kwargs


def make_autoencoder_figures(cfg, device, overwrite: bool, bfig_kwargs: dict):
    print("INFO: Creating autoencoder figures...")
    autoencoder = instantiate(cfg.models.galaxy_net)
    autoencoder.load_state_dict(torch.load(cfg.models.prior.galaxy_prior.autoencoder_ckpt))
    autoencoder = autoencoder.to(device).eval()

    # generate galsim simulated galaxies images if file does not exist.
    galaxies_file = Path(cfg.plots.sim_single_gals_file)
    if not galaxies_file.exists() or overwrite:
        print(f"INFO: Generating individual galaxy images and saving to: {galaxies_file}")
        dataset = instantiate(
            cfg.datasets.sdss_galaxies, batch_size=512, n_batches=20, num_workers=20
        )
        imagepath = galaxies_file.parent / (galaxies_file.stem + "_images.png")
        generate.generate(
            dataset, galaxies_file, imagepath, n_plots=25, global_params=("background", "slen")
        )

    # arguments for figures
    psf_params_file = cfg.plots.psf_params_file
    sdss_pixel_scale = cfg.plots.sdss_pixel_scale
    args = (autoencoder, galaxies_file, psf_params_file, sdss_pixel_scale)

    # create figure classes and plot.
    AutoEncoderReconRandom(n_examples=5, overwrite=overwrite, **bfig_kwargs)(*args)
    AutoEncoderBinMeasurements(overwrite=False, **bfig_kwargs)(*args)
    mpl.rc_file_defaults()


def make_blend_figures(cfg, encoder, decoder, overwrite: bool, bfig_kwargs: dict):
    print("INFO: Creating figures for metrics on simulated blended galaxies.")
    blend_file = Path(cfg.plots.sim_blend_gals_file)

    # create dataset of blends if not existant.
    if not blend_file.exists() or cfg.plots.overwrite:
        print(f"INFO: Creating dataset of simulated galsim blends and saving to {blend_file}")
        dataset = instantiate(cfg.plots.galsim_blends)
        imagepath = blend_file.parent / (blend_file.stem + "_images.png")
        global_params = ("background", "slen")
        generate.generate(dataset, blend_file, imagepath, n_plots=25, global_params=global_params)

    BlendGalsimFigure(overwrite=overwrite, **bfig_kwargs)(blend_file, encoder, decoder)


@hydra.main(config_path="./config", config_name="config", version_base=None)
def main(cfg):
    figs, device, bfig_kwargs = setup(cfg)
    encoder, decoder = load_models(cfg, device)
    overwrite = cfg.plots.overwrite

    # FIGURE 1: Autoencoder single galaxy reconstruction
    if "single_gal" in figs:
        make_autoencoder_figures(cfg, device, overwrite, bfig_kwargs)

    if "blend_gal" in figs:
        make_blend_figures(cfg, encoder, decoder, overwrite, bfig_kwargs)


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
