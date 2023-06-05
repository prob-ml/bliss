from typing import Dict, List, Optional

import numpy as np
import torch
from einops import rearrange, reduce
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import min_weight_full_bipartite_matching
from sklearn.metrics import confusion_matrix
from torch import Tensor
from torchmetrics import Metric

from bliss.catalog import FullCatalog


class BlissMetrics(Metric):
    """Calculates aggregate metrics on batches over full images (not tiles)."""

    # Detection metrics
    detection_tp: Tensor
    detection_fp: Tensor
    avg_distance: Tensor
    avg_keep_distance: Tensor
    total_true_n_souces: Tensor
    gal_tp: Tensor
    gal_fp: Tensor
    gal_fn: Tensor
    gal_tn: Tensor

    # Classification metrics
    disk_flux: List
    bulge_flux: List
    disk_q: List
    bulge_q: List
    disk_hlr: List
    bulge_hlr: List
    beta_radians: List

    full_state_update: Optional[bool] = True

    def __init__(
        self,
        slack: float = 1.0,
        dist_sync_on_step: bool = False,
        disable_bar: bool = True,
    ) -> None:
        """Computes matches between true and estimated locations.

        Args:
            slack: Threshold for matching objects a `slack` l-infinity distance away (in pixels).
            dist_sync_on_step: See torchmetrics documentation.
            disable_bar: Whether to show progress bar

        Attributes:
            detection_tp: true positives = # of sources matched with a true source.
            detection_fp: false positives = # of predicted sources not matched with true source
            avg_distance: Average l-infinity distance over all matched objects.
            avg_keep_distance: Average l-infinity distance over matched objects to keep.
            total_true_n_sources: Total number of true sources over batches seen.
            disable_bar: Whether to show progress bar
            gal_tp: true positives = # of sources correctly classified as a galaxy
            gal_fp: false positives = # of sources incorrectly classified as a galaxy
            gal_fn: false negatives = # of sources incorrectly classified as a star
            gal_tn: true negatives = # of sources correctly classified as a star
            disk_flux: Residuals of disk flux
            bulge_flux: Residuals of bulge flux
            disk_q: Residuals of disk ratio of major to minor axis
            bulge_q: Residuals of bulge ratio of major to minor axis
            disk_hlr: Residuals of disk half-light radius
            bulge_hlr: Residuals of bulge half-light radius
            beta_radians: Residuals of galaxy angle
        """
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.slack = slack
        self.disable_bar = disable_bar

        self.detection_metrics = [
            "detection_tp",
            "detection_fp",
            "avg_distance",
            "avg_keep_distance",
            "total_true_n_sources",
            "gal_tp",
            "gal_fp",
            "gal_fn",
            "gal_tn",
        ]
        for metric in self.detection_metrics:
            self.add_state(metric, default=torch.tensor(0.0), dist_reduce_fx="sum")

        self.classification_metrics = [
            "disk_flux",
            "bulge_flux",
            "disk_q",
            "bulge_q",
            "disk_hlr",
            "bulge_hlr",
            "beta_radians",
        ]
        for metric in self.classification_metrics:
            self.add_state(metric, default=[], dist_reduce_fx="sum")

    # pylint: disable=no-member
    def update(self, true: FullCatalog, est: FullCatalog) -> None:  # type: ignore
        """Update the internal state of metrics including tp, fp, total_coadd_gal_matches, etc."""
        assert true.batch_size == est.batch_size

        count, good_match_count = 0, 0
        for b in range(true.batch_size):
            ntrue, nest = true.n_sources[b].int().item(), est.n_sources[b].int().item()
            tlocs, elocs = true.plocs[b], est.plocs[b]
            tgbool, egbool = true["galaxy_bools"][b].reshape(-1), est["galaxy_bools"][b].reshape(-1)

            self.total_true_n_sources += ntrue  # type: ignore
            # if either ntrue or nest are 0, manually increment FP/FN and continue
            if ntrue == 0 or nest == 0:
                if nest > 0:  # all estimated are false positives
                    self.detection_fp += nest
                    self.gal_fp += egbool.sum()
                elif ntrue > 0:  # all true are false negatives
                    self.gal_fn += tgbool.sum()
                continue

            # Match true and estimated locations
            mtrue, mest, dkeep, avg_distance, avg_keep_distance = match_by_locs(
                tlocs[: int(ntrue)], elocs[: int(nest)], self.slack
            )

            # Compute detection metrics
            tgbool = tgbool[mtrue][dkeep].reshape(-1)
            egbool = egbool[mest][dkeep].reshape(-1)
            self._update_detection_metrics(
                elocs, nest, tgbool, egbool, mest, dkeep, avg_distance, avg_keep_distance
            )
            if not torch.isnan(avg_keep_distance):
                good_match_count += 1

            # Compute classification metrics
            if "galaxy_params" in true:
                true_gal_params = true["galaxy_params"][b][mtrue][dkeep]  # noqa: WPS529
                est_gal_params = est["galaxy_params"][b][mest][dkeep]
                self._update_galaxy_metrics(true_gal_params, est_gal_params)

            count += 1

        self.avg_distance /= count
        self.avg_keep_distance /= good_match_count

    def _update_detection_metrics(
        self, elocs, nest, tgbool, egbool, mest, dkeep, avg_distance, avg_keep_distance
    ) -> None:
        """Update detection metrics for a batch.

        Args:
            elocs: Locations of sources from estimated catalog.
            nest: Number of estimated sources.
            tgbool: Galaxy bools from true catalog.
            egbool: Galaxy bools from estimated catalog.
            mest: See match_by_locs.
            dkeep: See match_by_locs.
            avg_distance: See match_by_locs.
            avg_keep_distance: See match_by_locs.
        """
        detection_tp = len(elocs[mest][dkeep])
        detection_fp = nest - detection_tp
        assert detection_fp >= 0

        self.detection_tp += detection_tp
        self.detection_fp += detection_fp

        self.avg_distance += avg_distance
        # avg_keep_distance can be nan if no good matches were found, so ignore in that case
        if not torch.isnan(avg_keep_distance):
            self.avg_keep_distance += avg_keep_distance

        self._update_confusion_matrix(tgbool, egbool)  # compute confusion matrix

    def _update_confusion_matrix(self, tgbool: Tensor, egbool: Tensor) -> None:
        """Compute galaxy detection confusion matrix and update TP/FP/TN/FN.

        Args:
            tgbool (Tensor): Galaxy bools from true catalog.
            egbool (Tensor): Galaxy bools from estimated catalog.
        """
        conf_matrix = confusion_matrix(tgbool.cpu(), egbool.cpu(), labels=[1, 0])
        # decompose confusion matrix to pass to lightning logger
        self.gal_tp += conf_matrix[0][0]
        self.gal_fp += conf_matrix[0][1]
        self.gal_fn += conf_matrix[1][0]
        self.gal_tn += conf_matrix[1][1]

    def _update_galaxy_metrics(self, true_gal_params: Tensor, est_gal_params: Tensor) -> None:
        """Update galaxy classification metrics for a batch.

        Args:
            true_gal_params (Tensor): Galaxy parameters from true catalog.
                Parameters are total_flux, disk_frac, beta_radians, disk_q, a_d, bulge_q, a_b
            est_gal_params (Tensor): Galaxy parameters from estimated catalog.
        """
        # fluxes
        est_disk_flux = est_gal_params[:, 0] * est_gal_params[:, 1]  # total flux * disk fraction
        true_disk_flux = true_gal_params[:, 0] * true_gal_params[:, 1]
        self.disk_flux.append(torch.abs(true_disk_flux - est_disk_flux))

        est_bulge_flux = est_gal_params[:, 0] - est_disk_flux  # total flux - disk flux
        true_bulge_flux = true_gal_params[:, 0] - true_disk_flux
        self.bulge_flux.append(torch.abs(true_bulge_flux - est_bulge_flux))

        # angle
        self.beta_radians.append(torch.abs(true_gal_params[:, 2] - est_gal_params[:, 2]))

        # axis ratio
        self.disk_q.append(torch.abs(true_gal_params[:, 3] - est_gal_params[:, 3]))
        self.bulge_q.append(torch.abs(true_gal_params[:, 5] - est_gal_params[:, 5]))

        # half-light radius
        # sqrt(a * b) = sqrt(a * a * q) = a * sqrt(q)
        est_disk_hlr = est_gal_params[:, 4] * torch.sqrt(est_gal_params[:, 3])
        true_disk_hlr = true_gal_params[:, 4] * torch.sqrt(true_gal_params[:, 3])
        self.disk_hlr.append(torch.abs(true_disk_hlr - est_disk_hlr))

        est_bulge_hlr = est_gal_params[:, 6] * torch.sqrt(est_gal_params[:, 5])
        true_bulge_hlr = true_gal_params[:, 6] * torch.sqrt(true_gal_params[:, 5])
        self.bulge_hlr.append(torch.abs(true_bulge_hlr - est_bulge_hlr))

    def compute(self) -> Dict[str, Tensor]:
        """Calculate f1, misclassification accuracy, confusion matrix."""
        # PPV = positive predictive value
        det_precision = self.detection_tp / (self.detection_tp + self.detection_fp)
        # TPR = true positive rate
        det_recall = self.detection_tp / self.total_true_n_sources
        # f1 score
        f1 = (2 * det_precision * det_recall) / (det_precision + det_recall)
        # total number of predictions
        total_class = self.gal_tp + self.gal_fp + self.gal_tn + self.gal_fn

        metrics = {
            "detection_precision": det_precision,
            "detection_recall": det_recall,
            "f1": f1,
            "avg_distance": self.avg_distance,
            "avg_keep_distance": self.avg_keep_distance,
            "n_matches": self.gal_tp + self.gal_fp + self.gal_tn + self.gal_fn,
            "n_matches_gal_coadd": self.gal_tp + self.gal_fn,
            "class_acc": (self.gal_tp + self.gal_tn) / total_class,
            "gal_tp": self.gal_tp,
            "gal_fp": self.gal_fp,
            "gal_fn": self.gal_fn,
            "gal_tn": self.gal_tn,
        }

        # add classification metrics if computed
        if self.disk_flux:
            for metric in self.classification_metrics:
                # flatten list of variable-size tensors and take the median
                vals: List = sum([t.flatten().tolist() for t in getattr(self, metric)], [])
                mae = np.median(vals) if vals else np.nan
                metrics[f"{metric}_mae"] = torch.tensor(mae)

        return metrics


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
        - avg_distance: Average l-infinity distance over all matched objects.
        - avg_keep_distance: Average l-infinity distance over matched objects to keep.
    """
    assert len(true_locs.shape) == len(est_locs.shape) == 2
    assert true_locs.shape[-1] == est_locs.shape[-1] == 2
    assert isinstance(true_locs, torch.Tensor) and isinstance(est_locs, torch.Tensor)

    # reshape
    locs1 = true_locs.view(-1, 2)
    locs2 = est_locs.view(-1, 2)

    locs_abs_diff = (rearrange(locs1, "i j -> i 1 j") - rearrange(locs2, "i j -> 1 i j")).abs()
    locs_err = reduce(locs_abs_diff, "i j k -> i j", "sum")
    locs_err_l_infty = reduce(locs_abs_diff, "i j k -> i j", "max")

    # Penalize all pairs which are greater than slack apart to favor valid matches.
    locs_err = locs_err + (locs_err_l_infty > slack) * locs_err.max()

    # add small constant to avoid 0 weights (required for sparse bipartite matching)
    locs_err += 0.001

    # convert light source error matrix to CSR
    csr_locs_err = csr_matrix(locs_err.detach().cpu())

    # find minimal permutation and return matches
    row_indx, col_indx = min_weight_full_bipartite_matching(csr_locs_err)

    # only match objects that satisfy threshold on l-infinity distance.
    dist = (locs1[row_indx] - locs2[col_indx]).abs().max(1)[0]

    # GOOD match condition: L-infinity distance is less than slack
    dist_keep = (dist < slack).bool()
    avg_distance = dist.mean()
    avg_keep_distance = dist[dist < slack].mean()

    if dist_keep.sum() > 0:
        assert dist[dist_keep].max() <= slack

    return row_indx, col_indx, dist_keep, avg_distance, avg_keep_distance
