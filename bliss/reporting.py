"""Functions to evaluate the performance of BLISS predictions."""
from typing import Dict, Optional, Tuple

import galsim
import matplotlib as mpl
import numpy as np
import torch
import tqdm
from astropy.table import Table
from astropy.wcs import WCS
from einops import rearrange, reduce
from matplotlib.figure import Figure
from matplotlib.pyplot import Axes
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy import optimize as sp_optim
from sklearn.metrics import confusion_matrix
from torch import Tensor
from torchmetrics import Metric

from bliss.catalog import FullCatalog
from bliss.datasets.sdss import column_to_tensor, convert_flux_to_mag, convert_mag_to_flux


class DetectionMetrics(Metric):
    """Calculates aggregate detection metrics on batches over full images (not tiles)."""

    tp: Tensor
    fp: Tensor
    avg_distance: Tensor
    tp_gal: Tensor

    def __init__(
        self,
        slack=1.0,
        dist_sync_on_step=False,
    ) -> None:
        """Computes matches between true and estimated locations.

        Args:
            slack: Threshold for matching objects a `slack` l-infinity distance away (in pixels).
            dist_sync_on_step: See torchmetrics documentation.

        Attributes:
            tp: true positives = # of sources matched with a true source.
            fp: false positives = # of predicted sources not matched with true source
            avg_distance: Average l-infinity distance over matched objects.
            total_true_n_sources: Total number of true sources over batches seen.
            total_correct_class: Total # of correct classifications over matched objects.
            total_n_matches: Total # of matches over batches.
            conf_matrix: Confusion matrix (galaxy vs star)
        """
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.slack = slack

        self.add_state("tp", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("tp_gal", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("fp", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("avg_distance", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total_true_n_sources", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total_correct_class", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("conf_matrix", default=torch.tensor([[0, 0], [0, 0]]), dist_reduce_fx="sum")

    # pylint: disable=no-member
    def update(self, true: FullCatalog, est: FullCatalog):
        """Update the internal state of the metric including tp, fp, total_true_n_sources, etc."""
        assert isinstance(true, FullCatalog)
        assert isinstance(est, FullCatalog)
        assert true.batch_size == est.batch_size

        count = 0
        for b in range(true.batch_size):
            ntrue, nest = true.n_sources[b].int().item(), est.n_sources[b].int().item()
            tlocs, elocs = true.plocs[b], est.plocs[b]
            if ntrue > 0 and nest > 0:
                mtrue, mest, dkeep, avg_distance = match_by_locs(tlocs, elocs, self.slack)
                tp = len(elocs[mest][dkeep])  # n_matches
                true_galaxy_bools = true["galaxy_bools"][b][mtrue][dkeep]
                tp_gal = true_galaxy_bools.bool().sum()
                fp = nest - tp
                assert fp >= 0
                self.tp += tp
                self.tp_gal += tp_gal
                self.fp += fp
                self.avg_distance += avg_distance
                self.total_true_n_sources += ntrue
                count += 1
        self.avg_distance /= count

    def compute(self) -> Dict[str, Tensor]:
        precision = self.tp / (self.tp + self.fp)  # = PPV = positive predictive value
        recall = self.tp / self.total_true_n_sources  # = TPR = true positive rate
        f1 = (2 * precision * recall) / (precision + recall)
        return {
            "tp": self.tp,
            "fp": self.fp,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "avg_distance": self.avg_distance,
            "n_galaxies_detected": self.tp_gal,
        }


class ClassificationMetrics(Metric):
    """Calculates aggregate classification metrics on batches over full images (not tiles)."""

    total_n_matches: Tensor
    total_coadd_n_matches: Tensor
    total_coadd_gal_matches: Tensor
    total_correct_class: Tensor
    conf_matrix: Tensor

    def __init__(
        self,
        slack=1.0,
        dist_sync_on_step=False,
    ) -> None:
        """Computes matches between true and estimated locations.

        Args:
            slack: Threshold for matching objects a `slack` l-infinity distance away (in pixels).
            dist_sync_on_step: See torchmetrics documentation.

        Attributes:
            total_n_matches: Total number of true matches.
            total_correct_class: Total number of correct classifications.
            Confusion matrix: Confusion matrix of galaxy vs. star
        """
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.slack = slack

        self.add_state("total_n_matches", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total_coadd_gal_matches", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total_correct_class", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("conf_matrix", default=torch.tensor([[0, 0], [0, 0]]), dist_reduce_fx="sum")

    # pylint: disable=no-member
    def update(self, true, est):
        """Update the internal state of the metric including correct # of classifications."""
        assert isinstance(true, FullCatalog)
        assert isinstance(est, FullCatalog)
        assert true.batch_size == est.batch_size
        for b in range(true.batch_size):
            ntrue, nest = true.n_sources[b].int().item(), est.n_sources[b].int().item()
            tlocs, elocs = true.plocs[b], est.plocs[b]
            tgbool, egbool = true["galaxy_bools"][b].reshape(-1), est["galaxy_bools"][b].reshape(-1)
            if ntrue > 0 and nest > 0:
                mtrue, mest, dkeep, _ = match_by_locs(tlocs, elocs, self.slack)
                tgbool = tgbool[mtrue][dkeep].reshape(-1)
                egbool = egbool[mest][dkeep].reshape(-1)
                self.total_n_matches += len(egbool)
                self.total_coadd_gal_matches += tgbool.sum().int().item()
                self.total_correct_class += tgbool.eq(egbool).sum().int()
                self.conf_matrix += confusion_matrix(tgbool, egbool, labels=[1, 0])

    # pylint: disable=no-member
    def compute(self) -> Dict[str, Tensor]:
        """Calculate misclassification accuracy, and confusion matrix."""
        return {
            "n_matches": self.total_n_matches,
            "n_matches_gal_coadd": self.total_coadd_gal_matches,
            "class_acc": self.total_correct_class / self.total_n_matches,
            "conf_matrix": self.conf_matrix,
        }


def match_by_locs(true_locs, est_locs, slack=1.0):
    """Match true and estimated locations and returned indices to match.

    Permutes `est_locs` to find minimal error between `true_locs` and `est_locs`.
    The matching is done with `scipy.optimize.linear_sum_assignment`, which implements
    the Hungarian algorithm.

    Automatically discards matches where at least one location has coordinates **exactly** (0, 0).

    Args:
        slack: Threshold for matching objects a `slack` l-infinity distance away (in pixels).
        true_locs: Tensor of shape `(n1 x 2)`, where `n1` is the true number of sources.
            The centroids should be in units of PIXELS.
        est_locs: Tensor of shape `(n2 x 2)`, where `n2` is the predicted
            number of sources. The centroids should be in units of PIXELS.

    Returns:
        A tuple of the following objects:
        - row_indx: Indicies of true objects matched to estimated objects.
        - col_indx: Indicies of estimated objects matched to true objects.
        - dist_keep: Matched objects to keep based on l1 distances.
        - avg_distance: Average l-infinity distance over matched objects.
    """
    assert len(true_locs.shape) == len(est_locs.shape) == 2
    assert true_locs.shape[-1] == est_locs.shape[-1] == 2
    assert isinstance(true_locs, torch.Tensor) and isinstance(est_locs, torch.Tensor)

    locs1 = true_locs.view(-1, 2)
    locs2 = est_locs.view(-1, 2)

    locs_abs_diff = (rearrange(locs1, "i j -> i 1 j") - rearrange(locs2, "i j -> 1 i j")).abs()
    locs_err = reduce(locs_abs_diff, "i j k -> i j", "sum")
    locs_err_l_infty = reduce(locs_abs_diff, "i j k -> i j", "max")

    # Penalize all pairs which are greater than slack apart to favor valid matches.
    locs_err = locs_err + (locs_err_l_infty > slack) * locs_err.max()

    # find minimal permutation and return matches
    row_indx, col_indx = sp_optim.linear_sum_assignment(locs_err.detach().cpu())

    # we match objects based on distance too.
    # only match objects that satisfy threshold on l-infinity distance.
    # do not match fake objects with locs = (0, 0)
    dist = (locs1[row_indx] - locs2[col_indx]).abs().max(1)[0]
    origin_dist = torch.min(locs1[row_indx].pow(2).sum(1), locs2[col_indx].pow(2).sum(1))
    cond1 = (dist < slack).bool()
    cond2 = (origin_dist > 0).bool()
    dist_keep = torch.logical_and(cond1, cond2)
    avg_distance = dist[cond2].mean()  # average l-infinity distance over matched objects.
    if dist_keep.sum() > 0:
        assert dist[dist_keep].max() <= slack
    return row_indx, col_indx, dist_keep, avg_distance


def scene_metrics(
    true_params: FullCatalog,
    est_params: FullCatalog,
    mag_min: float = -np.inf,
    mag_max: float = np.inf,
    slack: float = 1.0,
):
    """Return detection and classification metrics based on a given ground truth.

    These metrics are computed as a function of magnitude based on the specified
    bin `(mag_min, mag_max)` but are designed to be independent of the estimated magnitude.
    Hence, precision is computed by taking a cut in the estimated parameters based on the magnitude
    bin and matching them with *any* true objects. Similarly, recall is computed by taking a cut
    on the true parameters and matching them with *any* predicted objects.

    Args:
        true_params: True parameters of each source in the scene (e.g. from coadd catalog)
        est_params: Predictions on scene obtained from predict_on_scene function.
        mag_min: Discard all objects with magnitude lower than this.
        mag_max: Discard all objects with magnitude higher than this.
        slack: Pixel L-infinity distance slack when doing matching for metrics.

    Returns:
        Dictionary with output from DetectionMetrics, ClassificationMetrics.
    """
    detection_metrics = DetectionMetrics(slack)
    classification_metrics = ClassificationMetrics(slack)

    # precision
    eparams = est_params.apply_mag_bin(mag_min, mag_max)
    detection_metrics.update(true_params, eparams)
    precision = detection_metrics.compute()["precision"]
    detection_metrics.reset()  # reset global state since recall and precision use different cuts.

    # recall
    tparams = true_params.apply_mag_bin(mag_min, mag_max)
    detection_metrics.update(tparams, est_params)
    recall = detection_metrics.compute()["recall"]
    n_galaxies_detected = detection_metrics.compute()["n_galaxies_detected"]
    detection_metrics.reset()

    # f1-score
    f1 = 2 * precision * recall / (precision + recall)
    detection_result = {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "n_galaxies_detected": n_galaxies_detected,
    }

    # classification
    tparams = true_params.apply_mag_bin(mag_min, mag_max)
    classification_metrics.update(tparams, est_params)
    classification_result = classification_metrics.compute()

    # report counts on each bin
    tparams = true_params.apply_mag_bin(mag_min, mag_max)
    eparams = est_params.apply_mag_bin(mag_min, mag_max)
    tcount = tparams.n_sources.int().item()
    tgcount = tparams["galaxy_bools"].sum().int().item()
    tscount = tcount - tgcount

    ecount = eparams.n_sources.int().item()
    egcount = eparams["galaxy_bools"].sum().int().item()
    escount = ecount - egcount

    n_matches = classification_result["n_matches"]
    n_matches_gal_coadd = classification_result["n_matches_gal_coadd"]

    counts = {
        "tgcount": tgcount,
        "tscount": tscount,
        "egcount": egcount,
        "escount": escount,
        "n_matches_coadd_gal": n_matches_gal_coadd,
        "n_matches_coadd_star": n_matches - n_matches_gal_coadd,
    }

    # compute and return results
    return {**detection_result, **classification_result, "counts": counts}


class CoaddFullCatalog(FullCatalog):
    coadd_names = {
        "objid": "objid",
        "galaxy_bools": "galaxy_bool",
        "fluxes": "flux",
        "mags": "mag",
        "ra": "ra",
        "dec": "dec",
    }
    allowed_params = FullCatalog.allowed_params.union(coadd_names.keys())

    @classmethod
    def from_file(
        cls, coadd_file: str, wcs: WCS, hlim: Tuple[int, int], wlim: Tuple[int, int], band="r"
    ):
        coadd_table = Table.read(coadd_file, format="fits")
        return cls.from_table(coadd_table, wcs, hlim, wlim, band)

    @classmethod
    def from_table(
        cls, cat, wcs: WCS, hlim: Tuple[int, int], wlim: Tuple[int, int], band: str = "r"
    ):
        """Load coadd catalog from file, add extra useful information, convert to tensors."""
        # filter saturated objects
        cat = cat[~cat["is_saturated"].data.astype(bool)]

        # add additional useful columns to coadd catalog
        x, y = wcs.all_world2pix(cat["ra"], cat["dec"], 0)
        galaxy_bools = ~cat["probpsf"].data.astype(bool)
        psfmag = cat[f"psfmag_{band}"] * cat["probpsf"]
        galmag = cat[f"modelMag_{band}"] * (1 - cat["probpsf"])
        mag = psfmag + galmag
        cat["x"] = x
        cat["y"] = y
        cat["galaxy_bool"] = galaxy_bools
        cat["mag"] = mag
        cat["flux"] = convert_mag_to_flux(mag)
        cat.replace_column("is_saturated", cat["is_saturated"].data.astype(bool))

        # misclassified bright galaxies in PHOTO as galaxies (obtaind by eye)
        misclass_ids = (8647475119820964111, 8647475119820964100, 8647475119820964192)
        for iid in misclass_ids:
            idx = np.where(cat["objid"] == iid)[0].item()
            cat["galaxy_bool"][idx] = 0

        # only return objects inside limits.
        w, h = cat["x"], cat["y"]
        keep = np.ones(len(cat)).astype(bool)
        keep &= (h > hlim[0]) & (h < hlim[1])
        keep &= (w > wlim[0]) & (w < wlim[1])
        height = hlim[1] - hlim[0]
        width = wlim[1] - wlim[0]
        data = {}
        h = torch.from_numpy(np.array(h).astype(np.float32)[keep])
        w = torch.from_numpy(np.array(w).astype(np.float32)[keep])

        # shift by +0.5 so it is consistent with BLISS parameters.
        data["plocs"] = torch.stack((h - hlim[0], w - wlim[0]), dim=1).unsqueeze(0) + 0.5
        data["n_sources"] = torch.tensor(data["plocs"].shape[1]).reshape(1)

        for bliss_name, coadd_name in cls.coadd_names.items():
            arr = column_to_tensor(cat, coadd_name)[keep]
            data[bliss_name] = rearrange(arr, "n_sources -> 1 n_sources 1")

        data["galaxy_bools"] = data["galaxy_bools"].bool()
        return cls(height, width, data)


def get_single_galaxy_measurements(
    slen: int,
    true_images: np.ndarray,
    recon_images: np.ndarray,
    psf_image: np.ndarray,
    pixel_scale: float = 0.396,
):
    """Compute individual galaxy measurements comparing true images with reconstructed images.

    Args:
        slen: Side-length of square input images.
        pixel_scale: Conversion from arcseconds to pixel.
        true_images: Array of shape (n_samples, n_bands, slen, slen) containing images of
            single-centered galaxies without noise or background.
        recon_images: Array of shape (n_samples, n_bands, slen, slen) containing
            reconstructions of `true_images` without noise or background.
        psf_image: Array of shape (n_bands, slen, slen) containing PSF image used for
            convolving the galaxies in `true_images`.

    Returns:
        Dictionary containing second-moment measurements for `true_images` and `recon_images`.
    """
    # TODO: Consider multiprocessing? (if necessary)
    assert true_images.shape == recon_images.shape
    assert len(true_images.shape) == len(recon_images.shape) == 4, "Incorrect array format."
    assert true_images.shape[1] == recon_images.shape[1] == psf_image.shape[0] == 1  # one band
    n_samples = true_images.shape[0]
    true_images = true_images.reshape(-1, slen, slen)
    recon_images = recon_images.reshape(-1, slen, slen)
    psf_image = psf_image.reshape(slen, slen)

    true_fluxes = true_images.sum(axis=(1, 2))
    recon_fluxes = recon_images.sum(axis=(1, 2))

    true_hlrs = np.zeros((n_samples))
    recon_hlrs = np.zeros((n_samples))
    true_ellip = np.zeros((n_samples, 2))  # 2nd shape: e1, e2
    recon_ellip = np.zeros((n_samples, 2))

    # get galsim PSF
    galsim_psf_image = galsim.Image(psf_image, scale=pixel_scale)

    # Now we use galsim to measure size and ellipticity
    for i in tqdm.tqdm(range(n_samples), desc="Measuring galaxies"):
        true_image = true_images[i]
        recon_image = recon_images[i]

        galsim_true_image = galsim.Image(true_image, scale=pixel_scale)
        galsim_recon_image = galsim.Image(recon_image, scale=pixel_scale)

        true_hlrs[i] = galsim_true_image.calculateHLR()  # PSF-convolved.
        recon_hlrs[i] = galsim_recon_image.calculateHLR()

        res_true = galsim.hsm.EstimateShear(
            galsim_true_image, galsim_psf_image, shear_est="KSB", strict=False
        )
        res_recon = galsim.hsm.EstimateShear(
            galsim_recon_image, galsim_psf_image, shear_est="KSB", strict=False
        )

        true_ellip[i, :] = (res_true.corrected_g1, res_true.corrected_g2)
        recon_ellip[i, :] = (res_recon.corrected_g1, res_recon.corrected_g2)

    return {
        "true_fluxes": true_fluxes,
        "recon_fluxes": recon_fluxes,
        "true_ellip": true_ellip,
        "recon_ellip": recon_ellip,
        "true_hlrs": true_hlrs,
        "recon_hlrs": recon_hlrs,
        "true_mags": convert_flux_to_mag(true_fluxes),
        "recon_mags": convert_flux_to_mag(recon_fluxes),
    }


def plot_image(
    fig: mpl.figure.Figure,
    ax: mpl.axes.Axes,
    image: np.ndarray,
    vrange: tuple = None,
    colorbar: bool = True,
    cmap="gray",
) -> None:
    assert len(image.shape) == 2
    vmin = image.min().item() if vrange is None else vrange[0]
    vmax = image.max().item() if vrange is None else vrange[1]

    if colorbar:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
    im = ax.matshow(image, vmin=vmin, vmax=vmax, cmap=cmap)
    if colorbar:
        fig.colorbar(im, cax=cax, orientation="vertical")


def plot_locs(
    ax: mpl.axes.Axes,
    bp: int,
    slen: int,
    plocs: np.ndarray,
    galaxy_probs: np.ndarray,
    m: str = "x",
    s: float = 20,
    lw: float = 1,
    alpha: float = 1,
    annotate=False,
    cmap: str = "RdYlBu",
) -> None:
    # NOTE: Only plot things inside border
    # NOTE: galaxy_probs can just be galaxy_bool.
    assert len(plocs.shape) == 2
    assert plocs.shape[1] == 2
    assert len(galaxy_probs.shape) == 1

    x = plocs[:, 1] - 0.5 + bp
    y = plocs[:, 0] - 0.5 + bp
    for i, (xi, yi) in enumerate(zip(x, y)):
        prob = galaxy_probs[i]
        cmp = mpl.cm.get_cmap(cmap)
        color = cmp(prob)
        if bp < xi < slen - bp and bp < yi < slen - bp:
            ax.scatter(xi, yi, color=color, marker=m, s=s, lw=lw, alpha=alpha)
            if annotate:
                ax.annotate(f"{galaxy_probs[i]:.2f}", (xi, yi), color=color, fontsize=8)


def plot_image_and_locs(
    fig: Figure,
    ax: Axes,
    idx: int,
    images: Tensor,
    bp: int,
    truth: Optional[FullCatalog] = None,
    estimate: Optional[FullCatalog] = None,
    vrange: tuple = None,
    s: float = 20,
    lw: float = 1,
    alpha: float = 1.0,
    labels: list = None,
    cmap_image: str = "gray",
    cmap_prob: str = "bwr",
    annotate_axis: bool = False,
    annotate_probs: bool = False,
    add_border: bool = False,
) -> None:
    # NOTE: labels must be a tuple/list of names with order (true star, true_gal, est_star, est_gal)
    # NOTE: true_plocs and est_plocs should be consistent both will be adjust with -0.5+bp
    assert len(images.shape) == 4, "Images should be batch form just like truth/estimate catalogs."
    assert images.shape[1] == 1, "Only 1 band supported."
    assert images.shape[-1] == images.shape[-2], "Only square images are supported."
    image = images[idx, 0].cpu().numpy()
    slen = images.shape[-1]

    # plot image first
    vmin = image.min().item() if vrange is None else vrange[0]
    vmax = image.max().item() if vrange is None else vrange[1]
    plot_image(fig, ax, image, vrange=(vmin, vmax), cmap=cmap_image)

    # (optionally) add white border showing where centers of stars and galaxies can be
    if add_border:
        ax.axvline(bp, color="w")
        ax.axvline(slen - bp, color="w")
        ax.axhline(bp, color="w")
        ax.axhline(slen - bp, color="w")

    if truth:
        # true parameters on full image.
        tplocs = truth.plocs[idx].cpu().numpy().reshape(-1, 2)
        tgbools = truth["galaxy_bools"][idx].float().cpu().numpy().reshape(-1)

        # plot true locations
        sp = s * 1.5
        plot_locs(ax, bp, slen, tplocs, tgbools, "+", s=sp, cmap="cool", alpha=alpha, lw=lw)

    if estimate is not None:
        n_sources = estimate.n_sources[idx].cpu().numpy().reshape(-1)
        plocs = estimate.plocs[idx].cpu().numpy().reshape(-1, 2)

        if annotate_axis is not None:
            assert truth is not None
            true_n_sources = truth.n_sources[idx].cpu().numpy()
            ax.set_xlabel(f"True num: {true_n_sources.item()}; Est num: {n_sources.item()}")

        gbools = estimate["galaxy_bools"].float().cpu().numpy().reshape(-1)
        gprobs = estimate.get("galaxy_probs", None)
        if annotate_probs:
            assert gprobs is not None
        gprobs = gbools if gprobs is None else gprobs.cpu().numpy().reshape(-1)

        plot_locs(
            ax, bp, slen, plocs, gprobs, "x", s, lw, alpha, annotate=annotate_probs, cmap=cmap_prob
        )

    if labels is not None:
        cmp1 = mpl.cm.get_cmap("cool")
        cmp2 = mpl.cm.get_cmap(cmap_prob)
        colors = (cmp1(1.0), cmp1(0.0), cmp2(1.0), cmp2(0.0))
        markers = ("+", "+", "x", "x")
        sizes = (s * 2, s * 2, s + 5, s + 5)

        if labels is not None:
            for label, c, m, size in zip(labels, colors, markers, sizes):
                ax.scatter([], [], color=c, marker=m, label=label, s=size)
            ax.legend(
                bbox_to_anchor=(0.0, 1.2, 1.0, 0.102),
                loc="lower left",
                ncol=2,
                mode="expand",
                borderaxespad=0.0,
            )
