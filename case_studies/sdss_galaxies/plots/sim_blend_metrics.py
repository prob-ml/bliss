from pathlib import Path

import numpy as np
import torch

from bliss.catalog import FullCatalog
from bliss.datasets import sdss
from bliss.encoder import Encoder
from bliss.models.decoder import ImageDecoder
from case_studies.sdss_galaxies.plots.bliss_figures import BlissFigures
from case_studies.sdss_galaxies.plots.sdss_detection_metrics import (
    compute_mag_bin_metrics,
    make_detection_figure,
)


class BlendSimFigures(BlissFigures):
    cache = "blendsim_cache.pt"

    def __init__(self, figdir, cachedir, overwrite=False, n_examples=5, img_format="png") -> None:
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
        plocs = blend_data["plocs"]
        fluxes = blend_data["params"][:, :, 0].reshape(n_batches, -1, 1)
        mags = sdss.convert_flux_to_mag(fluxes)
        full_catalog_dict = {"n_sources": n_sources, "plocs": plocs, "fluxes": fluxes, "mags": mags}
        full_truth = FullCatalog(slen, slen, full_catalog_dict)

        print("INFO: BLISS posterior inference on images.")

        batch_size = 128
        tile_est = None
        for jj in range(batch_size):
            images_jj = images[jj * batch_size : (jj + 1) * batch_size]
            background_jj = background[jj * batch_size : (jj + 1) * batch_size]
            tile_est_jj = encoder.variational_mode(images_jj, background_jj).cpu()

            if tile_est is None:
                tile_est = tile_est_jj
            else:
                tile_est = tile_est.cat
        tile_est.set_all_fluxes_and_mags(decoder)
        full_est = tile_est.to_full_params()

        print("INFO: Computing metrics")
        mag_cuts2 = np.arange(18, 24.5, 0.25)
        mag_cuts1 = np.full_like(mag_cuts2, fill_value=-np.inf)
        mag_cuts = np.column_stack((mag_cuts1, mag_cuts2))
        metrics = compute_mag_bin_metrics(mag_cuts, full_truth, full_est)

        return {"mag_bins": mag_cuts, "detection_metrics": metrics}

    def create_figures(self, data):
        mags = data["mag_bins"]
        detection_metrics = data["detection_metrics"]
        fig = make_detection_figure(mags, detection_metrics, xlims=(18, 24), ylims=(0, 1))
        return {"blend_detection_metrics": fig}
