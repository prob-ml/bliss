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
from bliss.reporting import (
    get_blendedness,
    get_deblended_reconstructions,
    get_residual_measurements,
)


# defaults correspond to 1 sigma in Gaussian distribution
def _calculate_statistics(y, x, x_bins, qs=(0.159, 0.841)):
    medians, q1s, q3s = [], [], []
    for ii in range(len(x_bins) - 1):
        _mask = (x > x_bins[ii]) * (x < x_bins[ii + 1])
        masked_y = y[_mask]
        medians.append(np.median(masked_y))
        q1s.append(np.quantile(masked_y, qs[0]))
        q3s.append(np.quantile(masked_y, qs[1]))
    return np.array(medians), np.array(q1s), np.array(q3s)


def _get_masked_data(data):
    # remove nans from data
    # use sigma as a reference
    sizes = data["truth"]["sigma"]
    bld_sizes = data["blended"]["sigma"]
    debld_sizes = data["deblended"]["sigma"]

    mask = ~(np.isnan(sizes) | np.isnan(bld_sizes) | np.isnan(debld_sizes))

    _snr = data["snr"][mask]
    _bld = data["bld"][mask]
    d1 = {k: v[mask] for k, v in data["truth"].items()}
    d2 = {k: v[mask] for k, v in data["deblended"].items()}
    d3 = {k: v[mask] for k, v in data["blended"].items()}

    return {"snr": _snr, "bld": _bld, "truth": d1, "deblended": d2, "blended": d3}


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
            "deblend_flux_scatter": {"fontsize": 32},
            "deblend_size_scatter": {"fontsize": 32},
            "deblend_ellips_scatter": {"fontsize": 32},
            "deblend_flux_bins": {"fontsize": 32},
            "deblend_size_bins": {"fontsize": 32},
            "deblend_ellips_bins": {"fontsize": 32},
        }

    @property
    def cache_name(self) -> str:
        return "deblend"

    @property
    def fignames(self) -> tuple[str, ...]:
        return (
            "deblend_flux_scatter",
            "deblend_size_scatter",
            "deblend_ellips_scatter",
            "deblend_flux_bins",
            "deblend_size_bins",
            "deblend_ellips_bins",
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
        # NOTE: Star fluxes are consistently remove from residual measurements as deblender
        # does not attempt to model stars and remove their contribution
        meas_truth = get_residual_measurements(
            _truth,
            images,
            paddings=paddings,
            sources=galaxy_uncentered,
            bp=bp,
            r=self.aperture,
            no_bar=False,
        )

        # get blendedness
        # NOTE: Here blendedness includes overlap with stars
        bld = get_blendedness(galaxy_uncentered, noiseless)

        # add parameters to truth
        _truth["fluxes_sep"] = meas_truth["flux"]
        _truth["snr"] = meas_truth["snr"]
        _truth["ellips"] = meas_truth["ellips"]
        _truth["sigma"] = meas_truth["sigma"]
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
        meas_not_deblended = get_residual_measurements(
            truth,
            images,
            paddings=paddings,
            sources=torch.zeros_like(uncentered_sources),
            bp=bp,
            r=self.aperture,
            no_bar=False,
        )
        fluxes_sep_bld = meas_not_deblended["flux"]
        ellips_sep_bld = meas_not_deblended["ellips"]
        sigma_sep_bld = meas_not_deblended["sigma"]

        # now for deblending case (on true locs)
        meas_deblended = get_residual_measurements(
            truth,
            images,
            paddings=paddings,
            sources=recon_uncentered,
            bp=bp,
            r=self.aperture,
            no_bar=False,
        )
        fluxes_sep_debld = meas_deblended["flux"]
        ellips_sep_debld = meas_deblended["ellips"]
        sigma_sep_debld = meas_deblended["sigma"]

        # now we organize the data to only keep snr > 0 sources that are galaxies
        # everything should be aligned
        mask = (truth["snr"].flatten() > 0) * (truth["galaxy_bools"].flatten() == 1)
        mask = mask.bool()

        # mask and flatten
        snr = truth["snr"].flatten()[mask]
        tfluxes = truth["fluxes_sep"].flatten()[mask]
        te1 = truth["ellips"][:, :, 0].flatten()[mask]
        te2 = truth["ellips"][:, :, 1].flatten()[mask]
        tsigma = truth["sigma"].flatten()[mask]

        bld_fluxes = fluxes_sep_bld.flatten()[mask]
        bld_e1 = ellips_sep_bld[:, :, 0].flatten()[mask]
        bld_e2 = ellips_sep_bld[:, :, 1].flatten()[mask]
        bld_sigma = sigma_sep_bld.flatten()[mask]

        debld_fluxes = fluxes_sep_debld.flatten()[mask]
        debld_e1 = ellips_sep_debld[:, :, 0].flatten()[mask]
        debld_e2 = ellips_sep_debld[:, :, 1].flatten()[mask]
        debld_sigma = sigma_sep_debld.flatten()[mask]

        bld = truth["blendedness"].flatten()[mask]

        return {
            "snr": snr,
            "bld": bld,
            "truth": {"flux": tfluxes, "e1": te1, "e2": te2, "sigma": tsigma},
            "blended": {"flux": bld_fluxes, "e1": bld_e1, "e2": bld_e2, "sigma": bld_sigma},
            "deblended": {
                "flux": debld_fluxes,
                "e1": debld_e1,
                "e2": debld_e2,
                "sigma": debld_sigma,
            },
        }

    def _get_deblending_flux_scatter(self, data):
        # first we make two scatter plot figures
        # useful for sanity checking

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

        snr = data["snr"]
        bld = data["bld"]
        fluxes = data["truth"]["flux"]
        bld_fluxes = data["blended"]["flux"]
        debld_fluxes = data["deblended"]["flux"]

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

    def _get_deblending_size_scatter(self, data):
        # first we make two scatter plot figures
        # useful for sanity checking

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        _data = _get_masked_data(data)

        snr = _data["snr"]
        bld = _data["bld"]
        sizes = _data["truth"]["sigma"]
        bld_sizes = _data["blended"]["sigma"]
        debld_sizes = _data["deblended"]["sigma"]

        res1 = (bld_sizes - sizes) / sizes
        res2 = (debld_sizes - sizes) / sizes

        ax1.scatter(snr, res1, marker="o", color="r", s=3)
        ax1.scatter(snr, res2, marker="o", color="b", s=3)
        ax1.set_xlim(1, 1000)
        ax1.set_xscale("log")
        ax1.set_ylim(-1, 5)

        ax2.scatter(bld, res1, marker="o", color="r", s=3)
        ax2.scatter(bld, res2, marker="o", color="b", s=3)
        ax2.set_xscale("log")
        ax2.set_xlim(1e-2, 1)
        ax2.set_ylim(-1, 5)

        plt.tight_layout()

        return fig

    def _get_deblending_ellips_scatter(self, data):
        # first we make two scatter plot figures
        # useful for sanity checking

        fig, axes = plt.subplots(2, 2, figsize=(20, 20))
        (ax1, ax2, ax3, ax4) = axes.ravel()
        _data = _get_masked_data(data)

        snr = _data["snr"]
        bld = _data["bld"]
        e1 = _data["truth"]["e1"]
        bld_e1 = _data["blended"]["e1"]
        debld_e1 = _data["deblended"]["e1"]
        e2 = _data["truth"]["e2"]
        bld_e2 = _data["blended"]["e2"]
        debld_e2 = _data["deblended"]["e2"]

        res1 = bld_e1 - e1
        res2 = debld_e1 - e1

        ax1.scatter(snr, res1, marker="o", color="r", s=3)
        ax1.scatter(snr, res2, marker="o", color="b", s=3)
        ax1.set_xlim(1, 1000)
        ax1.set_xscale("log")
        ax1.set_ylim(-1.5, 1.5)

        ax2.scatter(bld, res1, marker="o", color="r", s=3)
        ax2.scatter(bld, res2, marker="o", color="b", s=3)
        ax2.set_xscale("log")
        ax2.set_xlim(1e-2, 1)
        ax2.set_ylim(-1.5, 1.5)

        res1 = bld_e2 - e2
        res2 = debld_e2 - e2

        ax3.scatter(snr, res1, marker="o", color="r", s=3)
        ax3.scatter(snr, res2, marker="o", color="b", s=3)
        ax3.set_xlim(1, 1000)
        ax3.set_ylim(-1.5, 1.5)
        ax3.set_xscale("log")

        ax4.scatter(bld, res1, marker="o", color="r", s=3)
        ax4.scatter(bld, res2, marker="o", color="b", s=3)
        ax4.set_xlim(1e-2, 1)
        ax4.set_ylim(-1.5, 1.5)
        ax4.set_xscale("log")

        plt.tight_layout()

        return fig

    def _get_deblending_flux_bins(self, data):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

        snr = data["snr"]
        bld = data["bld"]
        fluxes = data["truth"]["flux"]
        bld_fluxes = data["blended"]["flux"]
        debld_fluxes = data["deblended"]["flux"]

        res1 = (bld_fluxes - fluxes) / fluxes
        res2 = (debld_fluxes - fluxes) / fluxes

        # equally sized bins in SNR
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

        # equally size bins in blendedness
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

    def _get_deblending_size_bins(self, data):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10), sharey=True)
        _data = _get_masked_data(data)

        snr = _data["snr"]
        bld = _data["bld"]
        sizes = _data["truth"]["sigma"]
        bld_sizes = _data["blended"]["sigma"]
        debld_sizes = _data["deblended"]["sigma"]

        res1 = (bld_sizes - sizes) / sizes
        res2 = (debld_sizes - sizes) / sizes

        # equally sized bins in SNR
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
        # ax1.set_ylim(-0.15, 0.2)
        ax1.set_xticks([3, 10, 100, 200])
        ax1.set_xlabel(r"\rm SNR")
        ax1.set_ylabel(r"$ (\sigma_{\rm pred} - \sigma_{\rm true}) / \sigma_{\rm true}$")
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
        # ax2.set_ylim(-0.75, 2.2)
        ax2.set_xlabel(r"\rm Blendedness")
        ax2.axhline(0.0, linestyle="--", color="k")

        plt.tight_layout()

        return fig

    def _get_deblending_ellips_bins(self, data):
        fig, axes = plt.subplots(2, 2, figsize=(20, 20))
        ax1, ax2, ax3, ax4 = axes.flatten()

        _data = _get_masked_data(data)

        snr = _data["snr"]
        bld = _data["bld"]
        e1 = _data["truth"]["e1"]
        bld_e1 = _data["blended"]["e1"]
        debld_e1 = _data["deblended"]["e1"]
        e2 = _data["truth"]["e2"]
        bld_e2 = _data["blended"]["e2"]
        debld_e2 = _data["deblended"]["e2"]

        res1 = bld_e1 - e1
        res2 = debld_e1 - e1

        # equally sized bins in SNR
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

        # ax1.set_ylim(-0.15, 0.2)
        ax1.set_xscale("log")
        ax1.set_xticks([3, 10, 100, 200])
        ax1.set_ylabel(r"$e_{1,\rm{pred}} -  e_{1,\rm{true}}$")
        ax1.axhline(0.0, linestyle="--", color="k")

        # equally sized bins in blendedness
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
        # ax2.set_ylim(-0.75, 2.2)
        ax2.axhline(0.0, linestyle="--", color="k")

        res1 = bld_e2 - e2
        res2 = debld_e2 - e2

        # equally sized bins in SNR
        snr_mask = (snr > 3) * (snr <= 1000)
        _snr = snr[snr_mask]
        _snr_log = np.log10(_snr)
        qs = np.linspace(0, 1, 10)
        snr_bins = np.quantile(_snr_log, qs)
        snr_middle = (snr_bins[1:] + snr_bins[:-1]) / 2

        meds1, qs11, qs12 = _calculate_statistics(res1[snr_mask], _snr_log, snr_bins)
        meds2, qs21, qs22 = _calculate_statistics(res2[snr_mask], _snr_log, snr_bins)

        ax3.plot(10**snr_middle, meds1, marker="o", color="r", label=r"\rm No Deblending")
        ax3.fill_between(10**snr_middle, qs11, qs12, color="r", alpha=0.5)

        ax3.plot(10**snr_middle, meds2, marker="o", color="b", label=r"\rm Deblending")
        ax3.fill_between(10**snr_middle, qs21, qs22, color="b", alpha=0.5)

        ax3.set_xscale("log")
        # ax3.set_ylim(-0.15, 0.2)
        ax3.set_xticks([3, 10, 100, 200])
        ax3.set_xlabel(r"\rm SNR")
        ax3.set_ylabel(r"$e_{2,\rm{pred}} -  e_{2,\rm{true}}$")
        ax3.axhline(0.0, linestyle="--", color="k")

        # equally sized bins in blendedness
        bld_mask = (bld > 1e-2) * (bld <= 1)
        _bld = bld[bld_mask]
        qs = np.linspace(0, 1, 12)
        bld_bins = np.quantile(_bld, qs)
        bld_middle = (bld_bins[1:] + bld_bins[:-1]) / 2

        meds1, qs11, qs12 = _calculate_statistics(res1[bld_mask], _bld, bld_bins)
        meds2, qs21, qs22 = _calculate_statistics(res2[bld_mask], _bld, bld_bins)

        ax4.plot(bld_middle, meds1, marker="o", color="r", label=r"\rm No Deblending")
        ax4.fill_between(bld_middle, qs11, qs12, color="r", alpha=0.5)

        ax4.plot(bld_middle, meds2, marker="o", color="b", label=r"\rm Deblending")
        ax4.fill_between(bld_middle, qs21, qs22, color="b", alpha=0.5)
        ax4.set_xscale("log")
        ax4.set_xticks([1e-2, 1e-1, 1])
        # ax2.set_ylim(-0.75, 2.2)
        ax4.set_xlabel(r"\rm Blendedness")
        ax4.axhline(0.0, linestyle="--", color="k")

        plt.tight_layout()

        return fig

    def create_figure(self, fname: str, data):
        if fname == "deblend_flux_scatter":
            return self._get_deblending_flux_scatter(data)
        if fname == "deblend_flux_bins":
            return self._get_deblending_flux_bins(data)
        if fname == "deblend_size_scatter":
            return self._get_deblending_size_scatter(data)
        if fname == "deblend_size_bins":
            return self._get_deblending_size_bins(data)
        if fname == "deblend_ellips_scatter":
            return self._get_deblending_ellips_scatter(data)
        if fname == "deblend_ellips_bins":
            return self._get_deblending_ellips_bins(data)
        raise ValueError(f"Unknown figure name: {fname}")
