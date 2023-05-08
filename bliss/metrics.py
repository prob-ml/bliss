from typing import Dict, Optional

import torch
from einops import rearrange, reduce
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import min_weight_full_bipartite_matching
from sklearn.metrics import confusion_matrix
from torch import Tensor
from torchmetrics import Metric
from tqdm import tqdm

from bliss.catalog import FullCatalog


class BlissMetrics(Metric):
    """Calculates aggregate detection metrics on batches over full images (not tiles)."""

    tp: Tensor
    fp: Tensor
    avg_distance: Tensor
    total_true_n_souces: Tensor
    gal_tp: Tensor
    gal_fp: Tensor
    gal_fn: Tensor
    gal_tn: Tensor
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
            tp: true positives = # of sources matched with a true source.
            fp: false positives = # of predicted sources not matched with true source
            avg_distance: Average l-infinity distance over matched objects.
            total_true_n_sources: Total number of true sources over batches seen.
            disable_bar: Whether to show progress bar
            gal_tp: true positives = # of sources correctly classified as a galaxy
            gal_fp: false positives = # of sources incorrectly classified as a galaxy
            gal_fn: false negatives = # of sources incorrectly classified as a star
            gal_tn: true negatives = # of sources correctly classified as a star
        """
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.slack = slack
        self.disable_bar = disable_bar

        self.add_state("tp", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("fp", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("avg_distance", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total_true_n_sources", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("gal_tp", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("gal_fp", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("gal_fn", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("gal_tn", default=torch.tensor(0), dist_reduce_fx="sum")

    # pylint: disable=no-member
    def update(self, true: FullCatalog, est: FullCatalog) -> None:  # type: ignore
        """Update the internal state of metrics including tp, fp, total_coadd_gal_matches, etc."""
        assert true.batch_size == est.batch_size

        count = 0
        desc = "Bliss Metric per batch"
        for b in tqdm(range(true.batch_size), desc=desc, disable=self.disable_bar):
            ntrue, nest = true.n_sources[b].int().item(), est.n_sources[b].int().item()
            tlocs, elocs = true.plocs[b], est.plocs[b]
            tgbool, egbool = true["galaxy_bools"][b].reshape(-1), est["galaxy_bools"][b].reshape(-1)
            if ntrue > 0 and nest > 0:
                mtrue, mest, dkeep, avg_distance = match_by_locs(tlocs, elocs, self.slack)
                tp = len(elocs[mest][dkeep])
                fp = nest - tp
                tgbool = tgbool[mtrue][dkeep].reshape(-1)
                egbool = egbool[mest][dkeep].reshape(-1)
                assert fp >= 0
                self.tp += tp
                self.fp += fp
                self.avg_distance += avg_distance
                self.total_true_n_sources += ntrue  # type: ignore
                conf_matrix = confusion_matrix(tgbool, egbool, labels=[1, 0])
                # decompose confusion matrix to pass to lightning logger
                self.gal_tp += conf_matrix[0][0]
                self.gal_fp += conf_matrix[0][1]
                self.gal_fn += conf_matrix[1][0]
                self.gal_tn += conf_matrix[1][1]

                count += 1
        self.avg_distance /= count

    def compute(self) -> Dict[str, Tensor]:
        """Calculate f1, misclassification accuracy, confusion matrix."""
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
            "n_matches": self.gal_tp + self.gal_fp + self.gal_tn + self.gal_fn,
            "n_matches_gal_coadd": self.gal_tp + self.gal_fn,
            "class_acc": (self.gal_tp + self.gal_tn)
            / (self.gal_tp + self.gal_fp + self.gal_tn + self.gal_fn),
            "gal_tp": self.gal_tp,
            "gal_fp": self.gal_fp,
            "gal_fn": self.gal_fn,
            "gal_tn": self.gal_tn,
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

    # reshape
    locs1 = true_locs.view(-1, 2)  # 11x2 - coordinates for true light sources
    locs2 = est_locs.view(-1, 2)  # 7x2 - coordinates for estimated light sources

    # remove non-existent estimated/true light sources
    locs1 = locs1[torch.abs(locs1).sum(dim=1) != 0]
    locs2 = locs2[torch.abs(locs2).sum(dim=1) != 0]

    locs_abs_diff = (rearrange(locs1, "i j -> i 1 j") - rearrange(locs2, "i j -> 1 i j")).abs()
    locs_err = reduce(locs_abs_diff, "i j k -> i j", "sum")
    locs_err_l_infty = reduce(locs_abs_diff, "i j k -> i j", "max")

    # Penalize all pairs which are greater than slack apart to favor valid matches.
    locs_err = locs_err + (locs_err_l_infty > slack) * locs_err.max()

    # convert light source error matrix to CSR
    csr_locs_err = csr_matrix(locs_err.detach().cpu())

    # find minimal permutation and return matches
    row_indx, col_indx = min_weight_full_bipartite_matching(csr_locs_err)

    # only match objects that satisfy threshold on l-infinity distance.
    dist = (locs1[row_indx] - locs2[col_indx]).abs().max(1)[0]

    # GOOD match condition: L-infinity distance is less than slack
    dist_keep = (dist < slack).bool()
    avg_distance = dist.mean()

    if dist_keep.sum() > 0:
        assert dist[dist_keep].max() <= slack

    return row_indx, col_indx, dist_keep, avg_distance
