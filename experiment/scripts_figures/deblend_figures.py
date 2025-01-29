"""Script to create detection encoder related figures."""

import math
from copy import deepcopy

import numpy as np
import torch
from einops import rearrange
from matplotlib import pyplot as plt
from tqdm import tqdm

from bliss.catalog import FullCatalog, TileCatalog
from bliss.datasets.io import load_dataset_npz
from bliss.encoders.deblend import GalaxyEncoder
from bliss.plotting import BlissFigure
from bliss.reporting import get_blendedness, get_deblended_reconstructions, get_fluxes_sep


# defaults correspond to 1 sigma in Gaussian distribution
def _calculate_statistics(residuals, x, x_bins, qs=(0.159, 0.841)):
    medians, q1s, q3s = [], [], []
    for ii in range(len(x_bins) - 1):
        _mask = (x > x_bins[ii]) * (x < x_bins[ii + 1])
        masked_residuals = residuals[_mask]
        medians.append(np.median(masked_residuals))
        q1s.append(np.quantile(masked_residuals, qs[0]))
        q3s.append(np.quantile(masked_residuals, qs[1]))
    return np.array(medians), np.array(q1s), np.array(q3s)


class DeblendingFigures(BlissFigure):
    def __init__(
        self, *, figdir, cachedir, suffix, overwrite=False, img_format="png", aperture=5.0
    ):
        super().__init__(
            figdir=figdir,
            cachedir=cachedir,
            suffix=suffix,
            overwrite=overwrite,
            img_format=img_format,
        )

        self.aperture = aperture

    @property
    def all_rcs(self) -> dict:
        return {
            "blends_deblending_scatter": {"fontsize": 32},
            "blends_deblending_bins": {"fontsize": 32},
        }

    @property
    def cache_name(self) -> str:
        return "deblend"

    @property
    def fignames(self) -> tuple[str, ...]:
        return (
            "blends_deblending_scatter",
            "blends_deblending_bins",
        )

    def compute_data(self, ds_path: str, deblend: GalaxyEncoder):

        # metadata
        bp = deblend.bp
        tile_slen = deblend.tile_slen
        ptile_slen = deblend.ptile_slen

        # read dataset
        dataset = load_dataset_npz(ds_path)
        images = dataset["images"]
        noiseless = dataset["noiseless"]
        paddings = dataset["paddings"]
        slen = images.shape[-1] - 2 * bp

        uncentered_sources = dataset["uncentered_sources"]
        galaxy_bools = dataset["galaxy_bools"]
        _tgbools = rearrange(galaxy_bools, "n ms 1 -> n ms 1 1 1 ")
        galaxy_uncentered = uncentered_sources * _tgbools

        # get truth catalog
        exclude = ("images", "uncentered_sources", "centered_sources", "noiseless", "paddings")
        true_cat_dict = {p: q for p, q in dataset.items() if p not in exclude}
        _truth = FullCatalog(slen, slen, true_cat_dict)

        # get fluxes through sep (only galaxies)
        # paddings includes stars, so their fluxes are removed
        fluxes, _, snr = get_fluxes_sep(
            _truth, images, paddings=paddings, sources=galaxy_uncentered, bp=bp, r=self.aperture
        )

        # get blendedness
        bld = get_blendedness(galaxy_uncentered, noiseless)

        # add parameters to truth
        _truth["fluxes_sep"] = fluxes
        _truth["snr"] = snr
        _truth["blendedness"] = bld.unsqueeze(-1)

        # we ignore double counting source and pick the brightest one for comparisons
        # these ensures results later are all aligned
        truth_tile_cat = _truth.to_tile_params(tile_slen, ignore_extra_sources=True)
        truth = truth_tile_cat.to_full_params()

        # run deblender using true centroids
        _batch_size = 50
        n_images = images.shape[0]
        n_batches = math.ceil(n_images / _batch_size)

        tiled_gparams = []

        for ii in tqdm(range(n_batches), desc="Encoding galaxy parameters"):
            start, end = ii * 50, (ii + 1) * 50
            bimages = images[start:end].to(deblend.device)
            btile_locs = truth_tile_cat.locs[start:end].to(deblend.device)
            _tile_gparams = deblend.variational_mode(bimages, btile_locs).to("cpu")
            tiled_gparams.append(_tile_gparams)

        tile_gparams = torch.concatenate(tiled_gparams, axis=0) * truth_tile_cat["galaxy_bools"]

        # create new catalog with these gparams, and otherwise true parameters
        est_tiled = deepcopy(truth_tile_cat.to_dict())
        est_tiled["galaxy_params"] = tile_gparams

        est_tiled_cat = TileCatalog(tile_slen, est_tiled)
        est = est_tiled_cat.to_full_params()

        assert est.batch_size == truth.batch_size == images.shape[0]

        # now we need to get full sized images with each individual galaxy, in the correct location.
        # we can actually cleverly zero out galaxy_bools to achieve this
        recon_uncentered = get_deblended_reconstructions(
            est,
            deblend._dec,
            slen=slen,
            ptile_slen=ptile_slen,
            tile_slen=tile_slen,
            device=deblend.device,
            bp=bp,
            no_bar=False,
        )

        # now get aperture fluxes for no deblending case (on true locs)
        fluxes_sep_bld, _, _ = get_fluxes_sep(
            truth,
            images,
            paddings=paddings,
            sources=torch.zeros_like(uncentered_sources),
            bp=bp,
            r=self.aperture,
        )

        # now for deblending case (on true locs)
        fluxes_sep_debld, _, _ = get_fluxes_sep(
            truth, images, paddings=paddings, sources=recon_uncentered, bp=bp, r=self.aperture
        )

        # now we organize the data to only keep snr > 0 sources that are galaxies
        # recall everything should be aligned
        mask = (truth["snr"].flatten() > 0) * (truth["galaxy_bools"].flatten() == 1)
        mask = mask.bool()
        snr = truth["snr"].flatten()[mask]
        tfluxes = truth["fluxes_sep"].flatten()[mask]
        debld_fluxes = fluxes_sep_debld.flatten()[mask]
        bld_fluxes = fluxes_sep_bld.flatten()[mask]
        bld = truth["blendedness"].flatten()[mask]

        return {
            "snr": snr,
            "bld": bld,
            "fluxes": tfluxes,
            "debld_fluxes": debld_fluxes,
            "bld_fluxes": bld_fluxes,
        }

    def _get_deblending_figure_scatter(self, data):
        # first we make two scatter plot figures
        # useful for sanity checking

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

        snr = data["snr"]
        bld = data["bld"]
        fluxes = data["fluxes"]
        bld_fluxes = data["bld_fluxes"]
        debld_fluxes = data["debld_fluxes"]

        res1 = (bld_fluxes - fluxes) / fluxes
        res2 = (debld_fluxes - fluxes) / fluxes

        ax1.scatter(snr, res1, marker="o", color="r", s=3)
        ax1.scatter(snr, res2, marker="o", color="b", s=3)
        ax1.set_xlim(1, 1000)
        ax1.set_xscale("log")
        ax1.set_ylim(-1, 10)

        ax2.scatter(bld, res1, marker="o", color="r", s=3)
        ax2.scatter(bld, res2, marker="o", color="b", s=3)
        ax2.set_xscale("log")
        ax2.set_xlim(1e-2, 1)
        ax2.set_ylim(-1, 10)

        plt.tight_layout()

        return fig

    def _get_deblending_figure_bins(self, data):
        # first we make two scatter plot figures
        # useful for sanity checking

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

        snr = data["snr"]
        bld = data["bld"]
        fluxes = data["fluxes"]
        bld_fluxes = data["bld_fluxes"]
        debld_fluxes = data["debld_fluxes"]

        res1 = (bld_fluxes - fluxes) / fluxes
        res2 = (debld_fluxes - fluxes) / fluxes

        # snr
        snr_mask = (snr > 3) * (snr <= 1000)
        _snr = snr[snr_mask]
        _snr_log = np.log10(_snr)
        qs = np.linspace(0, 1, 10)
        snr_bins = np.quantile(_snr_log, qs)
        snr_middle = (snr_bins[1:] + snr_bins[:-1]) / 2

        meds1, qs11, qs12 = _calculate_statistics(res1[snr_mask], _snr_log, snr_bins)
        meds2, qs21, qs22 = _calculate_statistics(res2[snr_mask], _snr_log, snr_bins)

        ax1.plot(10**snr_middle, meds1, marker="o", color="r", label=r"\rm No Deblending")
        ax1.fill_between(10**snr_middle, qs11, qs12, color="r", alpha=0.5)

        ax1.plot(10**snr_middle, meds2, marker="o", color="b", label=r"\rm Deblending")
        ax1.fill_between(10**snr_middle, qs21, qs22, color="b", alpha=0.5)

        ax1.set_xscale("log")
        ax1.set_ylim(-0.15, 0.2)
        ax1.set_xticks([3, 10, 100, 200])
        ax1.set_xlabel(r"\rm SNR")
        ax1.set_ylabel(r"$ (f_{\rm pred} - f_{\rm true}) / f_{\rm true}$")
        ax1.axhline(0.0, linestyle="--", color="k")

        # blendedness
        bld_mask = (bld > 1e-2) * (bld <= 1)
        _bld = bld[bld_mask]
        qs = np.linspace(0, 1, 12)
        bld_bins = np.quantile(_bld, qs)
        bld_middle = (bld_bins[1:] + bld_bins[:-1]) / 2

        meds1, qs11, qs12 = _calculate_statistics(res1[bld_mask], _bld, bld_bins)
        meds2, qs21, qs22 = _calculate_statistics(res2[bld_mask], _bld, bld_bins)

        ax2.plot(bld_middle, meds1, marker="o", color="r", label=r"\rm No Deblending")
        ax2.fill_between(bld_middle, qs11, qs12, color="r", alpha=0.5)

        ax2.plot(bld_middle, meds2, marker="o", color="b", label=r"\rm Deblending")
        ax2.fill_between(bld_middle, qs21, qs22, color="b", alpha=0.5)
        ax2.legend()
        ax2.set_xscale("log")
        ax2.set_xticks([1e-2, 1e-1, 1])
        ax2.set_ylim(-0.75, 2.2)
        ax2.set_xlabel(r"\rm Blendedness")
        ax2.axhline(0.0, linestyle="--", color="k")

        plt.tight_layout()

        return fig

    def create_figure(self, fname: str, data):
        if fname == "blends_deblending_scatter":
            return self._get_deblending_figure_scatter(data)
        if fname == "blends_deblending_bins":
            return self._get_deblending_figure_bins(data)
        raise ValueError(f"Unknown figure name: {fname}")
