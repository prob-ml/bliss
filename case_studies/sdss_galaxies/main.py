#!/usr/bin/env python3
"""Produce all figures. Save to nice PDF format."""
import os
from abc import abstractmethod
from pathlib import Path

import matplotlib as mpl
import numpy as np
import seaborn as sns
import torch
from astropy.table import Table
from astropy.wcs.wcs import WCS
from matplotlib import pyplot as plt

from bliss import reporting
from bliss.datasets import sdss
from bliss.datasets.galsim_galaxies import load_psf_from_file
from bliss.models.binary import BinaryEncoder
from bliss.models.galaxy_encoder import GalaxyEncoder
from bliss.predict import predict_on_scene
from bliss.sleep import SleepPhase

sns.set_theme(style="darkgrid")

device = torch.device("cuda:0")

files_dict = {
    "sleep_ckpt": "models/sdss_sleep.ckpt",
    "galaxy_encoder_ckpt": "models/sdss_galaxy_encoder.ckpt",
    "binary_ckpt": "models/sdss_binary.ckpt",
    "coadd_cat": "data/coadd_catalog_94_1_12.fits",
    "sdss_dir": "data/sdss",
    "psf_file": "data/psField-000094-1-0012-PSF-image.npy",
}


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
        "pixel_scale": 0.396,
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
    def __init__(self, outdir="", cache="temp.pt") -> None:
        os.chdir(os.getenv("BLISS_HOME"))
        outdir = Path(outdir)

        if not outdir.exists():
            outdir.mkdir(exist_ok=True)

        self.outdir = Path(outdir)
        self.cache = self.outdir / cache
        self.figures = {}

    @property
    @abstractmethod
    def fignames(self):
        """What figures will be produced with this class? What are their names?"""
        return {}

    def get_data(self, *args, **kwargs):
        """Return summary of data for producing plot, must be cachable w/ torch.save()."""
        if self.cache.exists():
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
            figs[k].savefig(fname, format="pdf")

    @abstractmethod
    def create_figures(self, data):
        """Return matplotlib figure instances to save based on data."""
        return mpl.figure.Figure()


class DetectionClassificationFigures(BlissFigures):
    def __init__(self, outdir="", cache="detect_class.pt") -> None:
        super().__init__(outdir=outdir, cache=cache)

    @property
    def fignames(self):
        return {
            "detection": self.outdir / "sdss-precision-recall.pdf",
            "classification": self.outdir / "sdss-classification-acc.pdf",
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


def main():
    os.chdir(os.getenv("BLISS_HOME"))  # simplicity for I/O

    # load data
    scene = get_sdss_data()["image"]
    coadd_cat = Table.read(files_dict["coadd_cat"], format="fits")

    # load models
    sleep_net = SleepPhase.load_from_checkpoint(files_dict["sleep_ckpt"]).to(device)
    binary_encoder = BinaryEncoder.load_from_checkpoint(files_dict["binary_ckpt"]).to(device).eval()
    galaxy_encoder = (
        GalaxyEncoder.load_from_checkpoint(files_dict["galaxy_encoder_ckpt"]).to(device).eval()
    )

    # FIGURE 1 : Classification and Detection metrics
    bfigure1 = DetectionClassificationFigures(outdir="case_studies/sdss_galaxies/output")
    bfigure1.save_figures(scene, coadd_cat, sleep_net, binary_encoder, galaxy_encoder)


if __name__ == "__main__":
    main()
