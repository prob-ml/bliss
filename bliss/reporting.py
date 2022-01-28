"""Functions to evaluate the performance of BLISS predictions."""
import galsim
import matplotlib as mpl
import numpy as np
import seaborn as sns
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

from bliss.datasets.sdss import convert_flux_to_mag, convert_mag_to_flux

CB_color_cycle = [
    "#377eb8",
    "#ff7f00",
    "#4daf4a",
    "#f781bf",
    "#a65628",
    "#984ea3",
    "#999999",
    "#e41a1c",
    "#dede00",
]


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
    def update(self, true_params: dict, est_params: dict):
        """Update the internal state of the metric including tp, fp, total_true_n_sources, etc."""
        true_n_sources, est_n_sources = true_params["n_sources"], est_params["n_sources"]
        true_locs, est_locs = true_params["plocs"], est_params["plocs"]  # plocs = pixel locs.
        assert len(true_n_sources.shape) == len(est_n_sources.shape) == 1, "Not tiles."
        assert true_n_sources.shape[0] == est_n_sources.shape[0]
        assert len(true_locs.shape) == len(est_locs.shape) == 3
        assert true_locs.shape[-1] == est_locs.shape[-1] == 2
        assert true_locs.shape[0] == est_locs.shape[0] == true_n_sources.shape[0]
        batch_size = true_n_sources.shape[0]

        count = 0
        for b in range(batch_size):
            ntrue, nest = true_n_sources[b].int().item(), est_n_sources[b].int().item()
            tlocs, elocs = true_locs[b], est_locs[b]
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
    def update(self, true_params: dict, est_params: dict):
        """Update the internal state of the metric including correct # of classifications."""
        true_n_sources, est_n_sources = true_params["n_sources"], est_params["n_sources"]
        true_locs, est_locs = true_params["plocs"], est_params["plocs"]
        true_galaxy_bool, est_galaxy_bool = true_params["galaxy_bool"], est_params["galaxy_bool"]
        batch_size = len(true_n_sources)
        assert len(true_galaxy_bool.shape) == len(est_galaxy_bool.shape) == 3
        assert true_galaxy_bool.shape[0] == est_galaxy_bool.shape[0] == batch_size
        assert len(true_locs.shape) == len(est_locs.shape) == 3
        assert true_locs.shape[-1] == est_locs.shape[-1] == 2
        assert true_locs.shape[0] == est_locs.shape[0] == batch_size

        for b in range(batch_size):
            ntrue, nest = true_n_sources[b].int().item(), est_n_sources[b].int().item()
            tlocs, elocs = true_locs[b], est_locs[b]
            tgbool, egbool = true_galaxy_bool[b].reshape(-1), est_galaxy_bool[b].reshape(-1)
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
    true_params: dict,
    est_params: dict,
    mag_cut=25.0,
    slack=1.0,
    mag_slack=0.25,
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
    tparams = apply_mag_cut(true_params, mag_cut + mag_slack)
    eparams = apply_mag_cut(est_params, mag_cut)

    # collect params in correct format
    tparams["plocs"] = tparams["plocs"].reshape(1, -1, 2)
    eparams["plocs"] = eparams["plocs"].reshape(1, -1, 2)

    # update
    detection_metrics.update(tparams, eparams)
    precision = detection_metrics.compute()["precision"]
    detection_metrics.reset()  # reset global state.
    assert detection_metrics.tp == 0  # pylint: disable=no-member

    # For calculating recall, we consider a wider bin for the 'estimated' objects, this way
    # we ensure that overestimating the magnitude of a true object close and below the boundary
    # does not mean that we missed this object, reducing our TP, and reducing recall.
    tparams = apply_mag_cut(true_params, mag_cut)
    eparams = apply_mag_cut(est_params, mag_cut + mag_slack)
    tparams["plocs"] = tparams["plocs"].reshape(1, -1, 2)
    eparams["plocs"] = eparams["plocs"].reshape(1, -1, 2)
    detection_metrics.update(tparams, eparams)
    recall = detection_metrics.compute()["recall"]
    detection_metrics.reset()

    # combine into f1 score and into single dictionary
    f1 = 2 * precision * recall / (precision + recall)
    detection_result = {"precision": precision, "recall": recall, "f1": f1}

    # compute classification metrics, these are only computed on matches so ignore mag_slack.
    tparams = apply_mag_cut(true_params, mag_cut)
    eparams = apply_mag_cut(est_params, mag_cut)
    tparams["plocs"] = tparams["plocs"].reshape(1, -1, 2)
    eparams["plocs"] = eparams["plocs"].reshape(1, -1, 2)
    tparams["galaxy_bool"] = tparams["galaxy_bool"].reshape(1, -1, 1)
    eparams["galaxy_bool"] = eparams["galaxy_bool"].reshape(1, -1, 1)
    classification_metrics.update(tparams, eparams)
    classification_result = classification_metrics.compute()

    # compute and return results
    return {**detection_result, **classification_result}


def apply_mag_cut(params: dict, mag_cut=25.0):
    """Apply magnitude cut to given parameters."""
    assert "mag" in params
    keep = params["mag"] < mag_cut
    d = {k: v[keep] for k, v in params.items() if k != "n_sources"}
    d["n_sources"] = torch.tensor([len(d["mag"])])
    return d


def get_params_from_coadd(coadd_cat: str, h: int, w: int, bp: int):
    """Load coadd catalog from file, add extra useful information, convert to tensors."""
    names = {"objid", "x", "y", "galaxy_bool", "flux", "mag", "hlr"}
    assert names.issubset(set(coadd_cat.columns))

    # filter saturated objects
    coadd_cat = coadd_cat[~coadd_cat["is_saturated"].data]

    # filter objects in border, outside of image, or that do not fit in a chunk.
    # NOTE: This assumes tiling scheme used in `predict.py`
    x, y = coadd_cat["x"], coadd_cat["y"]
    keep = (x > bp) & (x < w - bp) & (y > bp) & (y < h - bp)
    coadd_cat = coadd_cat[keep]

    # extract required arrays with correct byte order, otherwise torch error.
    data = {}
    for n in names:
        arr = []
        for v in coadd_cat[n]:
            arr.append(v)
        data[n] = torch.from_numpy(np.array(arr)).reshape(-1)

    # final adjustments.
    data["galaxy_bool"] = data["galaxy_bool"].bool()
    x, y = data["x"].reshape(-1, 1), data["y"].reshape(-1, 1)
    data["plocs"] = torch.hstack((x, y)).reshape(-1, 2)
    data["n_sources"] = len(x)

    return data


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


def plot_image(fig, ax, image, vrange=None):
    vmin = image.min().item() if vrange is None else vrange[0]
    vmax = image.max().item() if vrange is None else vrange[1]

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    im = ax.matshow(image, vmin=vmin, vmax=vmax, cmap="viridis")
    fig.colorbar(im, cax=cax, orientation="vertical")


def plot_locs(ax, slen, bpad, locs, color="r", marker="x", s=20, prob_galaxy=None):
    assert len(locs.shape) == 2
    assert locs.shape[1] == 2
    assert isinstance(slen, int)
    assert isinstance(bpad, int)
    if prob_galaxy is not None:
        assert len(prob_galaxy.shape) == 1

    x = locs[:, 1] * slen - 0.5 + bpad
    y = locs[:, 0] * slen - 0.5 + bpad
    for i, (xi, yi) in enumerate(zip(x, y)):
        if xi > bpad and yi > bpad:
            ax.scatter(xi, yi, color=color, marker=marker, s=s)
            if prob_galaxy is not None:
                ax.annotate(f"{prob_galaxy[i]:.2f}", (xi, yi), color=color, fontsize=8)


def plot_image_and_locs(
    idx: int,
    fig: Figure,
    ax: Axes,
    images,
    slen: int,
    true_params: dict,
    estimate: dict = None,
    labels: list = None,
    annotate_axis: bool = False,
    add_borders: bool = False,
    vrange: tuple = None,
    prob_galaxy: Tensor = None,
):
    # collect all necessary parameters to plot
    assert images.shape[1] == 1, "Only 1 band supported."
    if prob_galaxy is not None:
        assert "galaxy_bool" in estimate, "Inconsistent inputs to plot_image_and_locs"
    use_galaxy_bool = "galaxy_bool" in estimate if estimate is not None else False
    bpad = int((images.shape[-1] - slen) / 2)

    image = images[idx, 0].cpu().numpy()

    # true parameters on full image.
    true_n_sources = true_params["n_sources"][idx].cpu().numpy()
    true_locs = true_params["locs"][idx].cpu().numpy()
    true_galaxy_bool = true_params["galaxy_bool"][idx].cpu().numpy()
    true_star_bool = true_params["star_bool"][idx].cpu().numpy()
    true_galaxy_locs = true_locs * true_galaxy_bool
    true_star_locs = true_locs * true_star_bool

    # convert tile estimates to full parameterization for plotting
    if estimate is not None:
        n_sources = estimate["n_sources"][idx].cpu().numpy()
        locs = estimate["locs"][idx].cpu().numpy()

    if prob_galaxy is not None:
        prob_galaxy = prob_galaxy[idx].cpu().numpy().reshape(-1)

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
    plot_locs(ax, slen, bpad, true_galaxy_locs, "r", "x", s=20, prob_galaxy=None)
    plot_locs(ax, slen, bpad, true_star_locs, "c", "x", s=20, prob_galaxy=None)

    if estimate is not None:
        if use_galaxy_bool:
            galaxy_bool = estimate["galaxy_bool"][idx].cpu().numpy()
            star_bool = estimate["star_bool"][idx].cpu().numpy()
            galaxy_locs = locs * galaxy_bool
            star_locs = locs * star_bool
            plot_locs(ax, slen, bpad, galaxy_locs, "b", "+", s=30, prob_galaxy=prob_galaxy)
            plot_locs(ax, slen, bpad, star_locs, "m", "+", s=30, prob_galaxy=prob_galaxy)
        else:
            plot_locs(ax, slen, bpad, locs, "b", "+", s=30, prob_galaxy=None)

    if labels is not None:
        colors = ["r", "b", "c", "m"]
        markers = ["x", "+", "x", "+"]
        sizes = [25, 35, 25, 35]
        for l, c, m, s in zip(labels, colors, markers, sizes):
            if l is not None:
                ax.scatter(0, 0, color=c, s=s, marker=m, label=l)
        ax.legend(
            bbox_to_anchor=(0.0, 1.2, 1.0, 0.102),
            loc="lower left",
            ncol=2,
            mode="expand",
            borderaxespad=0.0,
        )


def set_rc_params(
    figsize=(10, 10),
    fontsize=18,
    title_size="large",
    label_size="medium",
    legend_fontsize="medium",
    tick_label_size="small",
    major_tick_size=7,
    minor_tick_size=4,
    major_tick_width=0.8,
    minor_tick_width=0.6,
    lines_marker_size=8,
):
    # named size options: 'xx-small', 'x-small', 'small', 'medium', 'large', 'x-large', 'xx-large'.
    rc_params = {
        # font.
        "font.family": "serif",
        "font.sans-serif": "Helvetica",
        "text.usetex": True,
        "mathtext.fontset": "cm",
        "font.size": fontsize,
        # figure
        "figure.figsize": figsize,
        # axes
        "axes.labelsize": label_size,
        "axes.titlesize": title_size,
        # ticks
        "xtick.labelsize": tick_label_size,
        "ytick.labelsize": tick_label_size,
        "xtick.major.size": major_tick_size,
        "ytick.major.size": major_tick_size,
        "xtick.major.width": major_tick_width,
        "ytick.major.width": major_tick_width,
        "ytick.minor.size": minor_tick_size,
        "xtick.minor.size": minor_tick_size,
        "xtick.minor.width": minor_tick_width,
        "ytick.minor.width": minor_tick_width,
        # markers
        "lines.markersize": lines_marker_size,
        # legend
        "legend.fontsize": legend_fontsize,
        # colors
        "axes.prop_cycle": mpl.cycler(color=CB_color_cycle),
    }
    mpl.rcParams.update(rc_params)
    sns.set_context(rc=rc_params)


def format_plot(ax, xlims=None, ylims=None, xticks=None, yticks=None, xlabel="", ylabel=""):
    if xlims is not None:
        ax.set_xlim(xlims)
    if ylims is not None:
        ax.set_ylim(ylims)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if xticks:
        ax.set_xticks(xticks)
    if yticks:
        ax.set_yticks(yticks)
