from pathlib import Path

import numpy as np
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm

from bliss.catalog import FullCatalog
from bliss.encoder import Encoder
from bliss.models.decoder import ImageDecoder
from bliss.reporting import match_by_locs
from case_studies.sdss_galaxies.plots.autoencoder import scatter_bin_plot, set_rc_params
from case_studies.sdss_galaxies.plots.bliss_figures import BlissFigures


class BlendSimFigures(BlissFigures):
    cache = "blendsim_cache.pt"

    def __init__(self, figdir, cachedir, overwrite=False, img_format="png") -> None:
        super().__init__(
            figdir=figdir, cachedir=cachedir, overwrite=overwrite, img_format=img_format
        )

    def compute_data(self, blend_file: Path, encoder: Encoder, decoder: ImageDecoder):
        blend_data = torch.load(blend_file)
        images = blend_data["images"]
        background = blend_data["background"]
        slen = blend_data["slen"].item()
        n_batches = images.shape[0]
        assert images.shape == (n_batches, 1, slen, slen)
        assert background.shape == (1, slen, slen)

        # prepare background
        background = background.unsqueeze(0)
        background = background.expand(n_batches, 1, slen, slen)

        # first create FullCatalog from simulated data
        print("INFO: Preparing full catalog from simulated blended sources.")
        n_sources = blend_data["n_sources"].reshape(-1)
        plocs = blend_data["plocs"].reshape(n_batches, -1, 2)
        fluxes = blend_data["fluxes"].reshape(n_batches, -1, 1)
        mags = blend_data["mags"].reshape(n_batches, -1, 1)
        ellips = blend_data["ellips"].reshape(n_batches, -1, 2)
        full_catalog_dict = {
            "n_sources": n_sources,
            "plocs": plocs,
            "fluxes": fluxes,
            "mags": mags,
            "ellips": ellips,
        }
        full_truth = FullCatalog(slen, slen, full_catalog_dict)

        print("INFO: BLISS posterior inference on images.")
        tile_est = encoder.variational_mode(images, background).cpu()
        tile_est.set_all_fluxes_and_mags(decoder)
        tile_est.set_galaxy_ellips(decoder, scale=0.393)
        full_est = tile_est.to_full_params()

        snr = []
        blendedness = []
        true_mags = []
        true_ellips = []
        est_mags = []
        est_ellips = []
        for ii in tqdm(range(n_batches), desc="Matching batches"):
            true_plocs, est_plocs = full_truth.plocs[ii], full_est.plocs[ii]

            tindx, eindx, dkeep = match_by_locs(true_plocs, est_plocs)
            n_matches = len(tindx[dkeep])

            snr_ii = blend_data["snr"][ii][tindx][dkeep]
            blendedness_ii = blend_data["blendedness"][ii][tindx][dkeep]
            true_mag_ii = full_truth["mags"][ii][tindx][dkeep]
            est_mag_ii = full_est["mags"][ii][eindx][dkeep]
            true_ellips_ii = full_truth["ellips"][ii][tindx][dkeep]
            est_ellips_ii = full_est["ellips"][ii][eindx][dkeep]
            n_matches = len(snr_ii)
            for jj in range(n_matches):
                snr.append(snr_ii[jj])
                blendedness.append(blendedness_ii[jj])
                true_mags.append(true_mag_ii[jj])
                true_ellips.append(true_ellips_ii[jj])
                est_mags.append(est_mag_ii[jj])
                est_ellips.append(est_ellips_ii[jj])

        # tensors
        snr = torch.tensor(snr)
        true_mags = torch.tensor(true_mags)

        return {
            "snr": torch.tensor(snr),
            "blendedness": torch.tensor(blendedness),
            "true_mags": torch.tensor(true_mags),
            "est_mags": torch.tensor(est_mags),
            "true_ellips": torch.tensor(true_ellips).reshape(-1, 2),
            "est_ellips": torch.tensor(est_ellips),
        }

    def make_snr_blendedness_bin_figure(self, data):
        snr, blendedness, true_mags, est_mags, true_ellips, est_ellips = data.values()
        fig, axes = plt.subplots(3, 2, figsize=(12, 18))
        ax1, ax2, ax3, ax4, ax5, ax6 = axes.flatten()
        set_rc_params(fontsize=24)

        xticks = [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
        xlims = (0, 3)
        xlabel = r"$\log_{10} \text{SNR}$"
        ylabel = r"\rm $m^{\rm recon} - m^{\rm true}$"

        x, y = np.log10(snr), est_mags - true_mags
        scatter_bin_plot(
            ax1,
            x,
            y,
            delta=0.2,
            xlims=xlims,
            xlabel=xlabel,
            ylabel=ylabel,
            xticks=xticks,
        )

        xticks = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
        xlims = (0, 1)
        xlabel = "$B$"
        ylabel = r"\rm $m^{\rm recon} - m^{\rm true}$"
        x, y = blendedness, est_mags - true_mags
        scatter_bin_plot(
            ax2,
            x,
            y,
            delta=0.2,
            xlims=xlims,
            xlabel=xlabel,
            ylabel=ylabel,
            xticks=xticks,
        )

        xticks = [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
        xlims = (0, 3)
        xlabel = r"$\log_{10} \text{SNR}$"
        ylabel = r"$g_{1}^{\rm recon} - g_{1}^{\rm true}$"
        x, y = np.log10(snr), est_ellips[:, 0] - true_ellips[:, 0]
        scatter_bin_plot(
            ax3,
            x,
            y,
            delta=0.2,
            xlims=xlims,
            xlabel=xlabel,
            ylabel=ylabel,
            xticks=xticks,
        )

        xticks = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
        xlims = (0, 1)
        xlabel = "$B$"
        ylabel = r"$g_{1}^{\rm recon} - g_{1}^{\rm true}$"
        x, y = blendedness, est_ellips[:, 0] - true_ellips[:, 0]
        scatter_bin_plot(
            ax4,
            x,
            y,
            delta=0.2,
            xlims=xlims,
            xlabel=xlabel,
            ylabel=ylabel,
            xticks=xticks,
        )

        xticks = [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
        xlims = (0, 3)
        xlabel = r"$\log_{10} \text{SNR}$"
        ylabel = r"$g_{2}^{\rm recon} - g_{2}^{\rm true}$"
        x, y = np.log10(snr), est_ellips[:, 1] - true_ellips[:, 1]
        scatter_bin_plot(
            ax5,
            x,
            y,
            delta=0.2,
            xlims=xlims,
            xlabel=xlabel,
            ylabel=ylabel,
            xticks=xticks,
        )

        xticks = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
        xlims = (0, 1)
        xlabel = "$B$"
        ylabel = r"$g_{2}^{\rm recon} - g_{2}^{\rm true}$"
        x, y = blendedness, est_ellips[:, 1] - true_ellips[:, 1]
        scatter_bin_plot(
            ax6,
            x,
            y,
            delta=0.2,
            xlims=xlims,
            xlabel=xlabel,
            ylabel=ylabel,
            xticks=xticks,
        )

        return fig

    def create_figures(self, data):
        return
        # fig = self.make_snr_blended_bin_figure(data)
        # return {"blend_detection_metrics": fig}
