#!/usr/bin/env python3
"""Produce all figures. Save to nice PDF format."""
import argparse
import os
import warnings
from abc import abstractmethod
from pathlib import Path

import matplotlib as mpl
import pytorch_lightning as pl
import torch
from astropy.table import Table

from bliss.datasets import sdss
from bliss.models.binary import BinaryEncoder
from bliss.models.galaxy_encoder import GalaxyEncoder
from bliss.sleep import SleepPhase
from bliss.case_studies.sdss_galaxies import reconstruction

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


def main(fig, outdir, overwrite=False):
    assert fig == "reconstruction", "Only SDSS reconstruction supported right now."
    os.chdir(os.getenv("BLISS_HOME"))  # simplicity for I/O

    if not Path(outdir).exists():
        warnings.warn("Specified output directory does not exist, will attempt to create it.")
        Path(outdir).mkdir(exist_ok=True, parents=True)

    sleep_net = SleepPhase.load_from_checkpoint(files_dict["sleep_ckpt"]).to(device)
    binary_encoder = BinaryEncoder.load_from_checkpoint(files_dict["binary_ckpt"])
    binary_encoder = binary_encoder.to(device).eval()
    galaxy_encoder = GalaxyEncoder.load_from_checkpoint(files_dict["galaxy_encoder_ckpt"])
    galaxy_encoder = galaxy_encoder.to(device).eval()

    # FIGURE 3: Reconstructions on SDSS
    sdss_data = get_sdss_data()
    coadd_cat = Table.read(files_dict["coadd_cat"], format="fits")
    sdss_rec_fig = SDSSReconstructionFigures(outdir, overwrite=overwrite)
    sdss_rec_fig.save_figures(sdss_data, coadd_cat, sleep_net, binary_encoder, galaxy_encoder)


def get_sdss_data(sdss_pixel_scale=0.396):
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
        "pixel_scale": sdss_pixel_scale,
    }


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
            figs[k].savefig(self.outdir / fname, format="pdf")

    @abstractmethod
    def create_figures(self, data):
        """Return matplotlib figure instances to save based on data."""
        return mpl.figure.Figure()


class SDSSReconstructionFigures(BlissFigures):
    def __init__(self, outdir="", cache="recon_sdss.pt", overwrite=False) -> None:
        super().__init__(outdir=outdir, cache=cache, overwrite=overwrite)

    @property
    def fignames(self):
        return {**{f"sdss_recon{i}": f"sdss_reconstruction{i}.pdf" for i in range(4)}}

    @property
    def lims(self):
        """Specificy spatial limits on frame to obtain chunks to reconstruct."""

        # NOTE: Decoder assumes square images.
        return {
            "sdss_recon0": ((1700, 2000), (200, 500)),  # scene
            "sdss_recon1": ((1000, 1300), (1150, 1450)),  # scene
            "sdss_recon2": ((742, 790), (460, 508)),  # individual blend
            "sdss_recon3": ((1128, 1160), (25, 57)),  # individual blend
            "sdss_recon4": ((500, 552), (170, 202)),  # individual blend
        }

    def compute_data(self, sdss_data, coadd_cat, sleep_net, binary_encoder, galaxy_encoder):
        data = {}
        for figname in self.fignames:
            lims = self.lims[figname]
            data[figname] = reconstruction.compute_data(
                sdss_data, coadd_cat, lims, sleep_net, binary_encoder, galaxy_encoder
            )
        return data

    def create_figures(self, data):
        """Make figures related to detection and classification in SDSS."""
        out_figures = {}

        for figname in self.fignames:
            data_fig = data[figname]
            out_figures[figname] = reconstruction.create_figure(data_fig)

        return out_figures


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create figures related to SDSS galaxies.")
    parser.add_argument(
        "-f",
        "--fig",
        help="Which figures do you want to create?",
        required=True,
        choices=["reconstruction"],
    )
    parser.add_argument("--overwrite", action="store_true", default=False, help="Recreate cache?")
    parser.add_argument(
        "-o",
        "--output",
        default="output/sdss_figures",
        type=str,
        help="Where to save figures and caches relative to $BLISS_HOME.",
    )
    args = vars(parser.parse_args())
    main(args["fig"], args["output"], args["overwrite"])
