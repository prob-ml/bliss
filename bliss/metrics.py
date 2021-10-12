import galsim
import numpy as np
import torch
import tqdm
from einops import rearrange, reduce
from scipy import optimize as sp_optim
from sklearn.metrics import confusion_matrix
from torchmetrics import Metric


def match_by_locs(true_locs, est_locs, slack=1.0):
    """Match true and estimated locations and returned indices to match.

    Permutes `est_locs` to find minimal error between `true_locs` and `est_locs`.
    The matching is done with `scipy.optimize.linear_sum_assignment`, which implements
    the Hungarian algorithm.

    Automatically discards leftover locations with coordinates **exactly** (0, 0).

    Args:
        slack: Threshold for matching objects a `slack` l-infinity distance away (in pixels).
        true_locs: Tensor of shape `(n1 x 2)`, where `n1` is the true number of sources.
            The centroids should be in units of pixels.
        est_locs: Tensor of shape `(n2 x 2)`, where `n2` is the predicted
            number of sources. The centroids should be in units of pixels.

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
    origin_dist = locs1[row_indx].pow(2).sum(1) + locs2[col_indx].pow(2).sum(1)  # avoid (0,0)
    cond1 = (dist < slack).bool()
    cond2 = (origin_dist > 0).bool()
    dist_keep = torch.logical_and(cond1, cond2)
    avg_distance = dist[cond2].mean()  # average l-infinity distance over matched objects.

    return row_indx, col_indx, dist_keep, avg_distance


class DetectionMetrics(Metric):
    """Class that calculates aggregate detection metrics over batches."""

    def __init__(
        self,
        slen,
        slack=1.0,
        dist_sync_on_step=False,
    ) -> None:
        """Computes matches between true and estimated locations.

        Args:
            slen: Size of image (w/out border paddding) in pixels.
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

        self.slen = slen
        self.slack = slack

        self.add_state("tp", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("fp", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("avg_distance", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total_true_n_sources", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total_n_matches", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total_correct_class", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("conf_matrix", default=torch.tensor([[0, 0], [0, 0]]), dist_reduce_fx="sum")

    # pylint: disable=no-member
    def update(self, true_params: dict, est_params: dict):
        """Update the internal state of the metric including tp, fp, total_true_n_sources, etc."""
        true_n_sources, est_n_sources = true_params["n_sources"], est_params["n_sources"]
        true_locs, est_locs = true_params["locs"], est_params["locs"]
        assert len(true_n_sources.shape) == len(est_n_sources.shape) == 1
        assert true_n_sources.shape[0] == est_n_sources.shape[0]
        assert len(true_locs.shape) == len(est_locs.shape) == 3
        assert true_locs.shape[-1] == est_locs.shape[-1] == 2
        assert true_locs.shape[0] == est_locs.shape[0] == true_n_sources.shape[0]
        batch_size = true_n_sources.shape[0]

        # get matches based on locations.
        true_locs *= self.slen
        est_locs *= self.slen

        self.total_true_n_sources += true_n_sources.sum().int().item()

        for b in range(batch_size):
            ntrue, nest = true_n_sources[b].int().item(), est_n_sources[b].int().item()
            if ntrue > 0 and nest > 0:
                _, mest, dkeep, avg_distance = match_by_locs(
                    true_locs[b], est_locs[b], slack=self.slack
                )
                elocs = est_locs[b][mest][dkeep]

                tp = len(elocs)
                fp = nest - len(elocs)
                assert fp >= 0
                self.tp += tp
                self.fp += fp
                self.total_n_matches += len(elocs)
                self.avg_distance += avg_distance
        self.avg_distance /= batch_size

    def compute(self):
        precision = self.tp / (self.tp + self.tp)  # = PPV = positive predictive value
        recall = self.tp / self.total_true_n_sources  # = TPR = true positive rate
        f1 = (2 * precision * recall) / (precision + recall)
        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "avg_distance": self.avg_distance,
        }


class ClassificationMetrics(Metric):
    """Class that calculates aggregate classification metrics over batches."""

    def __init__(
        self,
        slen,
        slack=1.0,
        dist_sync_on_step=False,
    ) -> None:
        """Computes matches between true and estimated locations.

        Args:
            slen: Side-length of image.
            slack: Threshold for matching objects a `slack` l-infinity distance away (in pixels).
            dist_sync_on_step: See torchmetrics documentation.

        Attributes:
            total_n_matches: Total number of true matches.
            total_correct_class: Total number of correct classifications.
            Confusion matrix: Confusion matrix of galaxy vs. star
        """
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.slen = slen
        self.slack = slack

        self.add_state("total_n_matches", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total_correct_class", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("conf_matrix", default=torch.tensor([[0, 0], [0, 0]]), dist_reduce_fx="sum")

    # pylint: disable=no-member
    def update(self, true_params: dict, est_params: dict):
        """Update the internal state of the metric including correct # of classifications."""
        true_n_sources, est_n_sources = true_params["n_sources"], est_params["n_sources"]
        true_locs, est_locs = true_params["locs"], est_params["locs"]
        true_galaxy_bool, est_galaxy_bool = true_params["galaxy_bool"], est_params["galaxy_bool"]
        batch_size = len(true_n_sources)
        assert len(true_galaxy_bool.shape) == len(est_galaxy_bool.shape) == 2
        assert true_galaxy_bool.shape[0] == est_galaxy_bool.shape[0] == batch_size
        assert len(true_locs.shape) == len(est_locs.shape) == 3
        assert true_locs.shape[-1] == est_locs.shape[-1] == 2
        assert true_locs.shape[0] == est_locs.shape[0] == batch_size

        # get matches based on locations.
        true_locs *= self.slen
        est_locs *= self.slen

        for b in range(batch_size):
            ntrue, nest = true_n_sources[b].int().item(), est_n_sources[b].int().item()
            if ntrue > 0 and nest > 0:
                mtrue, mest, dkeep, _ = match_by_locs(true_locs[b], est_locs[b], slack=self.slack)
                tgbool = true_galaxy_bool[b][mtrue][dkeep].reshape(-1)
                egbool = est_galaxy_bool[b][mest][dkeep].reshape(-1)
                self.total_n_matches += len(egbool)
                self.total_correct_class += tgbool.eq(egbool).sum().int()
                self.conf_matrix += confusion_matrix(tgbool, egbool)

    # pylint: disable=no-member
    def compute(self):
        """Calculate misclassification accuracy, and confusion matrix."""
        return {
            "class_acc": self.total_correct_class / self.total_n_matches,
            "conf_matrix": self.conf_matrix,
        }


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
    for i in tqdm.tqdm(range(n_samples)):
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
    }
