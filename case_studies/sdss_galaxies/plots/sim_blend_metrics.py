from pathlib import Path

import numpy as np
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm

from bliss.catalog import TileCatalog
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

    def make_snr_blendedness_bin_figure(self, data):
        set_rc_params(fontsize=24)
        snr, blendedness, true_mags, est_mags, true_ellips, est_ellips = data.values()
        fig, axes = plt.subplots(3, 2, figsize=(12, 18))
        ax1, ax2, ax3, ax4, ax5, ax6 = axes.flatten()

        xticks = [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
        xlims = (0, 3)
        xlabel = r"$\log_{10} \rm SNR$"
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
            delta=0.1,
            xlims=xlims,
            xlabel=xlabel,
            ylabel=ylabel,
            xticks=xticks,
        )

        xticks = [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
        xlims = (0, 3)
        xlabel = r"$\log_{10} \rm SNR$"
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
            delta=0.1,
            xlims=xlims,
            xlabel=xlabel,
            ylabel=ylabel,
            xticks=xticks,
        )

        xticks = [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
        xlims = (0, 3)
        xlabel = r"$\log_{10} \rm SNR$"
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
            delta=0.1,
            xlims=xlims,
            xlabel=xlabel,
            ylabel=ylabel,
            xticks=xticks,
        )

        plt.tight_layout()

        return fig

    def create_figures(self, data):
        fig = self.make_snr_blendedness_bin_figure(data)
        return {"blend_detection_metrics": fig}
