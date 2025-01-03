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
from bliss.render_tiles import reconstruct_image_from_ptiles, render_galaxy_ptiles
from bliss.reporting import get_blendedness, get_fluxes_sep


def _calculate_statistics(residuals, x, x_bins, qs=(0.25, 0.75)):
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
        fluxes, _, snr = get_fluxes_sep(
            _truth, images, paddings, galaxy_uncentered, bp, r=self.aperture
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

        # create new catalog with these gparams
        est_tiled = deepcopy(truth_tile_cat.to_dict())
        est_tiled["galaxy_params"] = tile_gparams

        est_tiled_cat = TileCatalog(tile_slen, est_tiled)
        est = est_tiled_cat.to_full_params()

        # now we need to get full sized images with each individual galaxy, in the correct location.
        # we can actually cleverly zero out galaxy_bools to achieve this
        n_images = images.shape[0]
        n_batches = math.ceil(n_images / _batch_size)
        recon_uncentered = torch.zeros(
            (n_images, est.max_n_sources, 1, images.shape[-2], images.shape[-1])
        )

        for jj in tqdm(range(est.max_n_sources), desc="Obtaining reconstructions"):
            # get mask
            mask = torch.arange(est.max_n_sources)
            mask = mask[mask != jj]

            # make a copy with all except one galaxy zeroed out
            est_jj = FullCatalog(slen, slen, deepcopy(est.to_dict()))
            est_jj["galaxy_bools"][:, mask, :] = 0
            est_jj["galaxy_bools"] = est_jj["galaxy_bools"].contiguous()
            est_tiled_jj = est_jj.to_tile_params(tile_slen, ignore_extra_sources=True)

            images_jj = []
            for kk in range(n_batches):
                start, end = kk * 50, (kk + 1) * 50
                blocs = est_tiled_jj.locs[start:end].to(deblend.device)
                bgparams = est_tiled_jj["galaxy_params"][start:end].to(deblend.device)
                bgbools = est_tiled_jj["galaxy_bools"][start:end].to(deblend.device)

                galaxy_tiles = render_galaxy_ptiles(
                    deblend._dec, blocs, bgparams, bgbools, ptile_slen, tile_slen
                ).to("cpu")

                galaxy_images = reconstruct_image_from_ptiles(galaxy_tiles, tile_slen)
                images_jj.append(galaxy_images)

            images_jj = torch.concatenate(images_jj, axis=0)
            recon_uncentered[:, jj, :, :, :] = images_jj

        # now get aperture fluxes for no deblending case (on true locs)
        n_images = images.shape[0]
        fluxes_sep_bld, _, _ = get_fluxes_sep(
            truth,
            images,
            paddings,
            torch.zeros_like(uncentered_sources),
            bp,
            r=self.aperture,
        )

        # now for deblending case (on true locs)
        fluxes_sep_debld, _, _ = get_fluxes_sep(
            truth, images, paddings, recon_uncentered, bp, r=self.aperture
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
        ax1.set_ylim(-0.025, 0.05)
        ax1.set_xticks([3, 10, 100, 200])
        ax1.set_xlabel(r"\rm SNR")
        ax1.set_ylabel(r"$ \Delta F / F$")

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
        ax2.set_ylim(-0.25, 1.5)
        ax2.set_xlabel(r"\rm Blendedness")

        plt.tight_layout()

        return fig

    def create_figure(self, fname: str, data):
        if fname == "blends_deblending_scatter":
            return self._get_deblending_figure_scatter(data)
        if fname == "blends_deblending_bins":
            return self._get_deblending_figure_bins(data)
        raise ValueError(f"Unknown figure name: {fname}")
