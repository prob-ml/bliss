"""Script to create detection encoder related figures."""

import math
from copy import deepcopy

import numpy as np
import torch
from einops import rearrange, reduce
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tqdm import tqdm

from bliss.catalog import FullCatalog, TileCatalog
from bliss.datasets.io import load_dataset_npz
from bliss.encoders.binary import BinaryEncoder
from bliss.plotting import CLR_CYCLE, BlissFigure
from bliss.render_tiles import get_n_padded_tiles_hw
from bliss.reporting import get_residual_measurements


def _get_metrics_per_bin(tbools, ebools, snrs, snr_bins):
    tp_per_bin = []
    nt_per_bin = []
    p_per_bin = []
    for ii in range(len(snr_bins) - 1):
        snr1 = snr_bins[ii]
        snr2 = snr_bins[ii + 1]

        _mask = (snrs > snr1) * (snrs < snr2)
        _tp = (ebools == tbools) * (tbools == 1) * (ebools == 1) * _mask
        _p = (ebools == 1) * _mask
        _nt = (tbools == 1) * _mask

        tp_per_bin.append(_tp.sum())
        p_per_bin.append(_p.sum())
        nt_per_bin.append(_nt.sum())

    tp = np.array(tp_per_bin)
    p = np.array(p_per_bin)
    nt = np.array(nt_per_bin)

    precision = tp / p
    recall = tp / nt
    f1 = 2 / (precision**-1 + recall**-1)

    return precision, recall, f1


def _get_equally_spaced_bins(
    bools: np.ndarray,
    snrs: np.ndarray,
    *,
    min_snr: float = 10.0,
    max_snr: float = 1000.0,
    n_bins: int = 10,
):
    mask = (snrs > min_snr) * (snrs <= max_snr) * bools.astype(bool)
    _log_snr = np.log10(snrs[mask])
    qs = np.linspace(0, 1, n_bins)
    snr_bins = 10 ** np.quantile(_log_snr, qs)
    snr_middle = (snr_bins[1:] + snr_bins[:-1]) / 2
    return snr_bins, snr_middle


class BinaryFigures(BlissFigure):
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
            "binary_scatter": {"fontsize": 34},
            "binary_curves": {"fontsize": 36, "major_tick_size": 12, "minor_tick_size": 7},
            "binary_contours": {"fontsize": 34},
        }

    @property
    def cache_name(self) -> str:
        return "binary"

    @property
    def fignames(self) -> tuple[str, ...]:
        return ("binary_scatter", "binary_curves", "binary_contours")

    def compute_data(self, ds_path: str, binary: BinaryEncoder):
        # metadata
        bp = binary.bp
        tile_slen = binary.tile_slen
        ptile_slen = binary.ptile_slen

        # read dataset
        dataset = load_dataset_npz(ds_path)
        images = dataset["images"]
        paddings = dataset["paddings"]
        uncentered_sources = dataset["uncentered_sources"]
        star_bools = dataset["star_bools"]
        size = images.shape[-1]
        slen = size - 2 * bp

        # paddings include stars for convenience, but we don't want to remove them in this case
        # we want to include snr of stars
        only_stars = uncentered_sources * rearrange(star_bools, "b n 1 -> b n 1 1 1").float()
        all_stars = reduce(only_stars, "b n c h w -> b c h w", "sum")
        new_paddings = paddings - all_stars

        # get truth catalog
        exclude = ("images", "uncentered_sources", "centered_sources", "noiseless", "paddings")
        true_cat_dict = {p: q for p, q in dataset.items() if p not in exclude}
        _truth = FullCatalog(slen, slen, true_cat_dict)

        # get snrs through sep
        meas_dict = get_residual_measurements(
            _truth,
            images,
            paddings=new_paddings,
            sources=uncentered_sources,
            bp=bp,
            r=self.aperture,
        )
        snr = meas_dict["snr"]

        # add parameters to truth
        _truth["snr"] = snr

        # we ignore double counting source and pick the brightest one for comparisons
        # these ensures results later are all aligned
        truth_tile_cat = _truth.to_tile_params(tile_slen, ignore_extra_sources=True)
        truth = truth_tile_cat.to_full_params()

        # get source is on
        b = truth.n_sources.shape[0]
        ms = truth.max_n_sources
        source_is_on = torch.zeros((b, ms))
        for jj in range(b):
            n = truth.n_sources[jj]
            source_is_on[jj, :n] = 1.0
        is_on_mask = source_is_on.flatten().bool()

        # run binary encoder on true locations
        batch_size = 100
        n_images = images.shape[0]
        n_batches = math.ceil(n_images / batch_size)
        nth, ntw = get_n_padded_tiles_hw(size, size, tile_slen=tile_slen, ptile_slen=ptile_slen)

        tile_galaxy_probs = []

        for ii in tqdm(range(n_batches)):
            start, end = ii * batch_size, (ii + 1) * batch_size
            bimages = images[start:end].to(binary.device)
            btile_locs = truth_tile_cat.locs[start:end].to(binary.device)
            tile_gprob_flat = binary.forward(bimages, btile_locs).to("cpu")
            tile_gprob = rearrange(
                tile_gprob_flat, "(n nth ntw) -> n nth ntw 1", n=batch_size, nth=nth, ntw=ntw
            )
            tile_galaxy_probs.append(tile_gprob)
        tile_galaxy_probs = torch.concatenate(tile_galaxy_probs, axis=0)

        # create new catalog with these booleans and prob
        out = {}
        thresholds = (0.5, 0.75, 0.9)
        for tsh in thresholds:
            est_tiled = deepcopy(truth_tile_cat.to_dict())
            n_sources_flat = est_tiled["n_sources"].float().unsqueeze(-1)
            est_tiled["galaxy_bools"] = tile_galaxy_probs.ge(tsh) * n_sources_flat
            est_tiled["star_bools"] = tile_galaxy_probs.le(1 - tsh) * n_sources_flat
            est_tiled["galaxy_probs"] = tile_galaxy_probs * n_sources_flat

            est_tiled_cat = TileCatalog(tile_slen, est_tiled)
            est = est_tiled_cat.to_full_params()

            # get flat list of truth, predicted bools, probs, and snr
            egbools = est["galaxy_bools"].flatten()[is_on_mask]
            esbools = est["star_bools"].flatten()[is_on_mask]
            probs = est["galaxy_probs"].flatten()[is_on_mask]

            out[tsh] = {
                "egbools": egbools,
                "esbools": esbools,
            }
            out["probs"] = probs  # always the same

        snr = truth["snr"].flatten()[is_on_mask]
        tgbools = truth["galaxy_bools"].flatten()[is_on_mask]
        tsbools = truth["star_bools"].flatten()[is_on_mask]

        out["snr"] = snr
        out["tgbools"] = tgbools
        out["tsbools"] = tsbools

        return out

    def _get_binary_scatter_figure(self, data: dict):
        # first we make two scatter plot figures
        # useful for sanity checking

        fig, ax = plt.subplots(1, 1, figsize=(12, 10))

        snr = data["snr"]
        probs = data["probs"]
        tgbools = data["tgbools"]
        tsbools = data["tsbools"]

        galaxy_mask = tgbools.astype(bool)
        star_mask = tsbools.astype(bool)

        c1 = CLR_CYCLE[0]
        c2 = CLR_CYCLE[1]

        # scatter plot of probabilities
        ax.scatter(
            snr[galaxy_mask],
            probs[galaxy_mask],
            marker="o",
            s=5,
            alpha=0.35,
            color=c1,
            label=r"\rm Galaxy",
        )
        ax.scatter(
            snr[star_mask],
            probs[star_mask],
            marker="o",
            s=5,
            alpha=0.35,
            color=c2,
            label=r"\rm Star",
        )
        ax.set_xscale("log")
        # ax.set_xticks([1e-2, 1e-1, 1, 10, 100, 1000, 10000, 100_000])
        ax.legend(markerscale=6, fontsize=28)
        ax.set_xlabel(r"\rm SNR")
        ax.set_ylabel(r"\rm Galaxy Classification Probability")
        ax.set_xlim(1, 1e3)

        return fig

    def _get_binary_curves(self, data: dict):
        snr = data["snr"]

        tgbools = data["tgbools"]
        tsbools = data["tsbools"]
        egbools = data[0.5]["egbools"]
        esbools = data[0.5]["esbools"]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 20))

        c1 = CLR_CYCLE[0]
        c2 = CLR_CYCLE[1]
        c3 = CLR_CYCLE[2]

        # precision and recall for galaxies
        snr_bins, snr_middle = _get_equally_spaced_bins(egbools, snr, n_bins=10)
        prec, rec, f1 = _get_metrics_per_bin(tgbools, egbools, snr, snr_bins)

        ax1.plot(snr_middle, prec, "-o", label=r"\rm precision", color=c1)
        ax1.plot(snr_middle, rec, "-o", label=r"\rm recall", color=c2)
        ax1.plot(snr_middle, f1, "-o", label="$F_{1}$", color=c3)
        ax1.set_xlabel(r"\rm SNR")
        ax1.set_xscale("log")
        ax1.set_title(r"\rm Galaxies")
        ax1.set_ylabel(r"\rm Metric", fontsize=36)
        ax1.set_xlim(10, 1000)
        ax1.legend()

        # precision and recall for stars
        snr_bins, snr_middle = _get_equally_spaced_bins(esbools, snr, n_bins=10)
        prec, rec, f1 = _get_metrics_per_bin(tsbools, esbools, snr, snr_bins)

        ax2.plot(snr_middle, prec, "-o", color=c1)
        ax2.plot(snr_middle, rec, "-o", color=c2)
        ax2.plot(snr_middle, f1, "-o", color=c3)
        ax2.set_xlabel(r"\rm SNR")
        ax2.set_xscale("log")
        ax2.set_title(r"\rm Stars")

        ax2.set_xlim(10, 1000)

        plt.tight_layout()

        return fig

    def _get_binary_contours(self, data: dict):
        snr = data["snr"]
        probs = data["probs"]
        tgbools = data["tgbools"]
        tsbools = data["tsbools"]

        galaxy_mask = tgbools.astype(bool) & (snr > 0)
        star_mask = tsbools.astype(bool) & (snr > 0)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10), sharey=True)

        ax1.hist2d(
            np.log10(snr[galaxy_mask]),
            probs[galaxy_mask],
            bins=20,
            range=[[0, 3], [0, 1]],
            cmap="PuBu",
            norm="log",
            vmin=1,
            vmax=2e4,
        )
        _xticks = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
        _xticks_labels = [f"$10^{int(x)}$" if x in [0.0, 1.0, 2.0, 3.0] else "" for x in _xticks]
        ax1.set_xticks(ticks=_xticks, labels=_xticks_labels)
        ax1.set_xlabel(r"\rm SNR")
        ax1.set_ylabel(r"\rm Galaxy Classification Probability")
        ax1.set_title(r"\rm Galaxies")

        _, _, _, pcm = ax2.hist2d(
            np.log10(snr[star_mask]),
            probs[star_mask],
            bins=20,
            range=[[0, 3], [0, 1]],
            cmap="PuBu",
            norm="log",
            vmin=1,
            vmax=2e4,
        )
        ax2.set_xticks(ticks=_xticks, labels=_xticks_labels)
        ax2.set_xlabel(r"\rm SNR")
        ax2.set_title(r"\rm Stars")

        divider = make_axes_locatable(ax2)
        cax = divider.append_axes("right", size="5%", pad=0.05)

        fig.colorbar(pcm, cax=cax, orientation="vertical")

        plt.tight_layout()

        return fig

    def create_figure(self, fname: str, data):
        if fname == "binary_scatter":
            return self._get_binary_scatter_figure(data)
        if fname == "binary_curves":
            return self._get_binary_curves(data)
        if fname == "binary_contours":
            return self._get_binary_contours(data)
        raise ValueError(f"Unknown figure name: {fname}")
