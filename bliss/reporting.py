"""Functions to evaluate the performance of BLISS predictions."""
from typing import Optional, Tuple

import galsim
import numpy as np
import torch
import tqdm
from astropy.table import Table
from einops import rearrange, reduce
from matplotlib.figure import Figure
from matplotlib.pyplot import Axes
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy import optimize as sp_optim
from sklearn.metrics import confusion_matrix
from torch import Tensor
from torchmetrics import Metric

from bliss.catalog import FullCatalog
from bliss.datasets.sdss import convert_flux_to_mag, convert_mag_to_flux


class DetectionMetrics(Metric):
    """Calculates aggregate detection metrics on batches over full images (not tiles)."""

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
                _, mest, dkeep, avg_distance = match_by_locs(tlocs, elocs, self.slack)
                tp = len(elocs[mest][dkeep])  # n_matches
                fp = nest - tp
                assert fp >= 0
                self.tp += tp
                self.fp += fp
                self.avg_distance += avg_distance
                self.total_true_n_sources += ntrue
                count += 1
        self.avg_distance /= count

    def compute(self):
        precision = self.tp / (self.tp + self.fp)  # = PPV = positive predictive value
        recall = self.tp / self.total_true_n_sources  # = TPR = true positive rate
        f1 = (2 * precision * recall) / (precision + recall)
        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "avg_distance": self.avg_distance,
        }


class ClassificationMetrics(Metric):
    """Calculates aggregate classification metrics on batches over full images (not tiles)."""

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
        self.add_state("total_correct_class", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("conf_matrix", default=torch.tensor([[0, 0], [0, 0]]), dist_reduce_fx="sum")

    # pylint: disable=no-member
    def update(self, true: FullCatalog, est: FullCatalog):
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
                self.total_correct_class += tgbool.eq(egbool).sum().int()
                self.conf_matrix += confusion_matrix(tgbool, egbool, labels=[1, 0])

    # pylint: disable=no-member
    def compute(self):
        """Calculate misclassification accuracy, and confusion matrix."""
        return {
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
        dist_keep: Matched objects to keep based on l1 distances.
        distances: Average l-infinity distance over matched objects.
        match_true, match_est: Tensors corresponding to indices matched, so
            that `true_locs[true_match]`, `est_locs[est_match]` will have the matched
            locations for each pair of matched objects at the same position.
    """
    assert len(true_locs.shape) == len(est_locs.shape) == 2
    assert true_locs.shape[-1] == est_locs.shape[-1] == 2

    locs1 = true_locs.view(-1, 2)
    locs2 = est_locs.view(-1, 2)

    # entry (i,j) is l1 distance between of ith loc in locs1 and the jth loc in locs2
    locs_err = (rearrange(locs1, "i j -> i 1 j") - rearrange(locs2, "i j -> 1 i j")).abs()
    locs_err = reduce(locs_err, "i j k -> i j", "sum")

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

    return row_indx, col_indx, dist_keep, avg_distance


def scene_metrics(
    true_params: FullCatalog,
    est_params: FullCatalog,
    mag_cut=25.0,
    slack=1.0,
    mag_slack=1.0,
):
    """Metrics based on using the coadd catalog as truth.

    We apply the given magnitude cut to both the coadd and estimated objects, and only consider
    objects brighter than a certain magnitude. Additionally a slack in the estimated magnitude
    is used so that objects close to the magnitude boundary do not negatively affect performance.

    Args:
        true_params: True parameters of each source in the scene (e.g. from coadd catalog)
        est_params: Predictions on scene obtained from predict_on_scene function.
        mag_cut: Magnitude cut, discard all objects with magnitude higher than this.
        slack: Pixel L-infinity distance slack when doing matching for metrics.
        mag_slack: Consider objects above mag_cut for precision/recall to avoid edge effects.

    Returns:
        Dictionary with output from DetectionMetrics, ClassificationMetrics.
    """

    # prepare metrics
    detection_metrics = DetectionMetrics(slack)
    classification_metrics = ClassificationMetrics(slack)

    # For calculating precision, we consider a wider bin for the 'true' objects, this way
    # we ensure that underestimating the magnitude of a true object close and above the
    # boundary does not mark this estimated object as FP, thus reducing our precision.
    tparams = true_params.apply_mag_cut(mag_cut + mag_slack)
    eparams = est_params.apply_mag_cut(mag_cut)

    # update
    detection_metrics.update(tparams, eparams)
    precision = detection_metrics.compute()["precision"]
    detection_metrics.reset()  # reset global state.
    assert detection_metrics.tp == 0  # pylint: disable=no-member

    # For calculating recall, we consider a wider bin for the 'estimated' objects, this way
    # we ensure that overestimating the magnitude of a true object close and below the boundary
    # does not mean that we missed this object, reducing our TP, and reducing recall.
    tparams = true_params.apply_mag_cut(mag_cut)
    eparams = est_params.apply_mag_cut(mag_cut + mag_slack)
    detection_metrics.update(tparams, eparams)
    recall = detection_metrics.compute()["recall"]
    detection_metrics.reset()

    # combine into f1 score and into single dictionary
    f1 = 2 * precision * recall / (precision + recall)
    detection_result = {"precision": precision, "recall": recall, "f1": f1}

    # compute classification metrics, these are only computed on matches so ignore mag_slack.
    tparams = true_params.apply_mag_cut(mag_cut)
    eparams = est_params.apply_mag_cut(mag_cut)
    classification_metrics.update(tparams, eparams)
    classification_result = classification_metrics.compute()

    # compute and return results
    return {**detection_result, **classification_result}


class CoaddFullCatalog(FullCatalog):
    coadd_names = {
        "objid": "objid",
        "galaxy_bools": "galaxy_bool",
        "fluxes": "flux",
        "mags": "mag",
        "hlr": "hlr",
        "ra": "ra",
        "dec": "dec",
    }
    allowed_params = FullCatalog.allowed_params.union(coadd_names.keys())

    @classmethod
    def from_file(cls, coadd_file: str, hlim: Tuple[int, int], wlim: Tuple[int, int]):
        coadd_table = Table.read(coadd_file, format="fits")
        return cls.from_table(coadd_table, hlim, wlim)

    @classmethod
    def from_table(
        cls,
        coadd_table,
        hlim: Tuple[int, int],
        wlim: Tuple[int, int],
    ):
        """Load coadd catalog from file, add extra useful information, convert to tensors."""
        assert set(cls.coadd_names.values()).issubset(set(coadd_table.columns))
        # filter saturated objects
        coadd_table = coadd_table[~coadd_table["is_saturated"].data]
        # only return objects inside limits.
        w, h = coadd_table["x"], coadd_table["y"]
        keep = np.ones(len(coadd_table)).astype(bool)
        keep &= (h > hlim[0]) & (h < hlim[1])
        keep &= (w > wlim[0]) & (w < wlim[1])
        height = hlim[1] - hlim[0]
        width = wlim[1] - wlim[0]
        data = {}
        h = torch.from_numpy(np.array(h).astype(np.float32)[keep])
        w = torch.from_numpy(np.array(w).astype(np.float32)[keep])
        data["plocs"] = torch.stack((h - hlim[0], w - wlim[0]), dim=1).unsqueeze(0)
        data["n_sources"] = torch.tensor(data["plocs"].shape[1]).reshape(1)

        for bliss_name, coadd_name in cls.coadd_names.items():
            arr = column_to_tensor(coadd_table, coadd_name)[keep]
            data[bliss_name] = rearrange(arr, "n_sources -> 1 n_sources 1")
        data["galaxy_bools"] = data["galaxy_bools"].bool()
        return cls(height, width, data)


def column_to_tensor(table, colname):
    dtypes = {
        np.dtype(">i8"): int,
        np.dtype("bool"): bool,
        np.dtype(">f8"): np.float32,
        np.dtype("float64"): np.dtype("float64"),
    }
    x = np.array(table[colname])
    dtype = dtypes[x.dtype]
    x = x.astype(dtype)
    return torch.from_numpy(x)


def get_flux_coadd(coadd_cat, nelec_per_nmgy=987.31, band="r"):
    """Get flux and magnitude measurements for a given SDSS Coadd catalog."""
    fluxes = []
    mags = []
    for entry in coadd_cat:
        is_star = bool(entry["probpsf"])
        if is_star:
            psfmag = entry[f"psfmag_{band}"]
            flux = convert_mag_to_flux(psfmag, nelec_per_nmgy)
            mag = psfmag
        else:  # is galaxy
            devmag = entry[f"devmag_{band}"]
            expmag = entry[f"expmag_{band}"]
            devflux = convert_mag_to_flux(devmag, nelec_per_nmgy)
            expflux = convert_mag_to_flux(expmag, nelec_per_nmgy)
            flux = devflux + expflux
            mag = convert_flux_to_mag(flux, nelec_per_nmgy)

        fluxes.append(flux)
        mags.append(mag)

    return np.array(fluxes), np.array(mags)


def get_hlr_coadd(coadd_cat: Table, psf: galsim.GSObject, nelec_per_nmgy: float = 987.31):
    if "hlr" in coadd_cat.colnames:
        return coadd_cat["hlr"]

    hlrs = []
    psf_hlr = psf.calculateHLR()
    for entry in tqdm.tqdm(coadd_cat, desc="Calculating HLR"):

        is_star = bool(entry["probpsf"])
        if is_star:
            hlrs.append(psf_hlr)
        else:
            components = []
            disk_flux = convert_mag_to_flux(entry["expmag_r"], nelec_per_nmgy)
            bulge_flux = convert_mag_to_flux(entry["devmag_r"], nelec_per_nmgy)

            if disk_flux > 0:
                disk_beta = np.radians(entry["expphi_r"])  # radians
                disk_hlr = entry["exprad_r"]  # arcsecs
                disk_q = entry["expab_r"]
                disk = galsim.Exponential(flux=disk_flux, half_light_radius=disk_hlr)
                disk = disk.shear(q=disk_q, beta=disk_beta * galsim.radians)
                components.append(disk)

            if bulge_flux > 0:
                bulge_beta = np.radians(entry["devphi_r"])
                bulge_hlr = entry["devrad_r"]
                bulge_q = entry["devab_r"]
                bulge = galsim.DeVaucouleurs(flux=bulge_flux, half_light_radius=bulge_hlr)
                bulge = bulge.shear(q=bulge_q, beta=bulge_beta * galsim.radians)
                components.append(bulge)
            gal = galsim.Add(components)
            gal = galsim.Convolution(gal, psf)
            try:
                hlr = gal.calculateHLR()
            except galsim.errors.GalSimFFTSizeError:
                hlr = np.nan
            hlrs.append(hlr)
    return np.array(hlrs)


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


def plot_image(fig, ax, image, vrange=None, colorbar=True, cmap="viridis"):
    vmin = image.min().item() if vrange is None else vrange[0]
    vmax = image.max().item() if vrange is None else vrange[1]

    if colorbar:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
    im = ax.matshow(image, vmin=vmin, vmax=vmax, cmap=cmap)
    if colorbar:
        fig.colorbar(im, cax=cax, orientation="vertical")


def plot_locs(ax, bpad, plocs, color="r", marker="x", s=20, galaxy_probs=None):
    assert len(plocs.shape) == 2
    assert plocs.shape[1] == 2
    assert isinstance(bpad, int)
    if galaxy_probs is not None:
        assert len(galaxy_probs.shape) == 1

    x = plocs[:, 1] - 0.5 + bpad
    y = plocs[:, 0] - 0.5 + bpad
    for i, (xi, yi) in enumerate(zip(x, y)):
        if xi > bpad and yi > bpad:
            ax.scatter(xi, yi, color=color, marker=marker, s=s)
            if galaxy_probs is not None:
                ax.annotate(f"{galaxy_probs[i]:.2f}", (xi, yi), color=color, fontsize=8)


def plot_image_and_locs(
    idx: int,
    fig: Figure,
    ax: Axes,
    images,
    slen: int,
    true_params: FullCatalog,
    estimate: Optional[FullCatalog] = None,
    labels: list = None,
    annotate_axis: bool = False,
    add_borders: bool = False,
    vrange: tuple = None,
    galaxy_probs: Optional[Tensor] = None,
):
    # collect all necessary parameters to plot
    assert images.shape[1] == 1, "Only 1 band supported."
    if galaxy_probs is not None:
        assert "galaxy_bools" in estimate, "Inconsistent inputs to plot_image_and_locs"
    use_galaxy_bools = "galaxy_bools" in estimate if estimate is not None else False
    bpad = int((images.shape[-1] - slen) / 2)

    image = images[idx, 0].cpu().numpy()

    # true parameters on full image.
    true_n_sources = true_params.n_sources[idx].cpu().numpy()
    true_plocs = true_params.plocs[idx].cpu().numpy()
    true_galaxy_bools = true_params["galaxy_bools"][idx].cpu().numpy()
    true_star_bools = true_params["star_bools"][idx].cpu().numpy()
    true_galaxy_plocs = true_plocs * true_galaxy_bools
    true_star_plocs = true_plocs * true_star_bools

    # convert tile estimates to full parameterization for plotting
    if estimate is not None:
        n_sources = estimate.n_sources[idx].cpu().numpy()
        plocs = estimate.plocs[idx].cpu().numpy()

    if galaxy_probs is not None:
        galaxy_probs = galaxy_probs[idx].cpu().numpy().reshape(-1)

    # annotate useful information around the axis
    if annotate_axis and estimate is not None:
        ax.set_xlabel(f"True num: {true_n_sources.item()}; Est num: {n_sources.item()}")

    # (optionally) add white border showing where centers of stars and galaxies can be
    if add_borders:
        ax.axvline(bpad, color="w")
        ax.axvline(bpad + slen, color="w")
        ax.axhline(bpad, color="w")
        ax.axhline(bpad + slen, color="w")

    # plot image first
    vmin = image.min().item() if vrange is None else vrange[0]
    vmax = image.max().item() if vrange is None else vrange[1]
    plot_image(fig, ax, image, vrange=(vmin, vmax))

    # plot locations
    plot_locs(ax, bpad, true_galaxy_plocs, "r", "x", s=20, galaxy_probs=None)
    plot_locs(ax, bpad, true_star_plocs, "c", "x", s=20, galaxy_probs=None)

    if estimate is not None:
        if use_galaxy_bools:
            galaxy_bools = estimate["galaxy_bools"][idx].cpu().numpy()
            star_bools = estimate["star_bools"][idx].cpu().numpy()
            galaxy_plocs = plocs * galaxy_bools
            star_plocs = plocs * star_bools
            plot_locs(ax, bpad, galaxy_plocs, "b", "+", s=30, galaxy_probs=galaxy_probs)
            plot_locs(ax, bpad, star_plocs, "m", "+", s=30, galaxy_probs=galaxy_probs)
        else:
            plot_locs(ax, bpad, plocs, "b", "+", s=30, galaxy_probs=None)

    if labels is not None:
        colors = ["r", "b", "c", "m"]
        markers = ["x", "+", "x", "+"]
        sizes = [25, 35, 25, 35]
        for ell, c, m, s in zip(labels, colors, markers, sizes):
            if ell is not None:
                ax.scatter(0, 0, color=c, s=s, marker=m, label=ell)
        ax.legend(
            bbox_to_anchor=(0.0, 1.2, 1.0, 0.102),
            loc="lower left",
            ncol=2,
            mode="expand",
            borderaxespad=0.0,
        )
