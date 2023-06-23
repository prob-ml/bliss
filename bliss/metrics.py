# pylint: disable=E1101
# mypy: disable-error-code="union-attr"

from enum import Enum
from typing import Dict, List, Union

import numpy as np
import torch
from einops import rearrange, reduce
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import min_weight_full_bipartite_matching
from sklearn.metrics import confusion_matrix
from torch import Tensor
from torchmetrics import Metric

from bliss.catalog import FullCatalog, SourceType, TileCatalog

# define type alias to simplify signatures
Catalog = Union[TileCatalog, FullCatalog]


class MetricsMode(Enum):
    FULL = 1
    TILE = 2


class BlissMetrics(Metric):
    """Calculates detection and classification metrics between two catalogs.

    BlissMetrics supports two modes, Full and Tile, which indicate what type of catalog to operate
    over. For FullCatalogs, all metrics are computed by matching predicted sources to true sources.
    For TileCatalogs, detection metrics are still computed that way, but star/galaxy parameter
    metrics are computed conditioned on true source location and type; i.e, given a true star or
    galaxy in a tile in the true catalog, get the corresponding parameters in the same tile in the
    estimated catalog (regardless of type/existence predicted by estimated catalog in that tile.

    Note that galaxy classification metrics are only computed when the values are available in
    both catalogs.
    """

    # detection metrics
    detection_tp: Tensor
    detection_fp: Tensor
    total_true_n_sources: Tensor
    total_distance: Tensor
    total_distance_keep: Tensor
    match_count: Tensor
    match_count_keep: Tensor
    gal_tp: Tensor
    gal_fp: Tensor
    star_tp: Tensor
    star_fp: Tensor

    # classification metrics
    star_flux: List
    disk_flux: List
    bulge_flux: List
    disk_q: List
    bulge_q: List
    disk_hlr: List
    bulge_hlr: List
    beta_radians: List

    full_state_update: bool = False

    def __init__(
        self,
        mode: MetricsMode = MetricsMode.FULL,
        slack: float = 1.0,
        dist_sync_on_step: bool = False,
        disable_bar: bool = True,
    ) -> None:
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.slack = slack
        self.disable_bar = disable_bar
        self.mode = mode

        self.detection_metrics = [
            "detection_tp",
            "detection_fp",
            "total_true_n_sources",
            "total_distance",
            "total_distance_keep",
            "match_count",
            "match_count_keep",
            "gal_tp",
            "gal_fp",
            "star_tp",
            "star_fp",
        ]
        for metric in self.detection_metrics:
            self.add_state(metric, default=torch.tensor(0.0), dist_reduce_fx="sum")

        self.classification_metrics = [
            "star_flux",
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

    def update(self, true: Catalog, est: Catalog) -> None:
        assert true.batch_size == est.batch_size
        if self.mode is MetricsMode.FULL:
            msg = "Metrics mode is set to `Full` but received a TileCatalog"
            assert isinstance(true, FullCatalog) and isinstance(est, FullCatalog), msg
        elif self.mode is MetricsMode.TILE:
            msg = "Metrics mode is set to `Tile` but received a FullCatalog"
            assert isinstance(true, TileCatalog) and isinstance(est, TileCatalog), msg

        match_true, match_est = self._update_detection_metrics(true, est)
        self._update_classification_metrics(true, est, match_true, match_est)

    def _update_detection_metrics(self, true: Catalog, est: Catalog) -> None:
        """Update detection metrics."""
        if self.mode is MetricsMode.FULL:
            true_locs = true.plocs
            est_locs = est.plocs
            tgbool = true.galaxy_bools
            egbool = est.galaxy_bools
        elif self.mode is MetricsMode.TILE:
            true_on_idx, true_is_on = true.get_indices_of_on_sources()
            est_on_idx, est_is_on = est.get_indices_of_on_sources()

            true_locs = true.gather_param_at_tiles("locs", true_on_idx)
            true_locs *= true_is_on.unsqueeze(-1)
            est_locs = est.gather_param_at_tiles("locs", est_on_idx)
            est_locs *= est_is_on.unsqueeze(-1)

            true_st = true.gather_param_at_tiles("source_type", true_on_idx)
            tgbool = true_st == SourceType.GALAXY
            tgbool *= true_is_on.unsqueeze(-1)

            est_st = est.gather_param_at_tiles("source_type", est_on_idx)
            egbool = est_st == SourceType.GALAXY
            egbool *= est_is_on.unsqueeze(-1)

        match_true, match_est = [], []
        for b in range(true.batch_size):
            ntrue = int(true.n_sources[b].int().sum().item())
            nest = int(est.n_sources[b].int().sum().item())
            self.total_true_n_sources += ntrue

            # if either ntrue or nest are 0, manually increment FP/FN and continue
            if ntrue == 0 or nest == 0:
                if nest > 0:
                    self.detection_fp += nest
                continue

            mtrue, mest, dkeep, avg_distance, avg_keep_distance = match_by_locs(
                true_locs[b, 0:ntrue], est_locs[b, 0:nest], self.slack
            )
            match_true.append(mtrue[dkeep])
            match_est.append(mest[dkeep])

            # update TP/FP and distances
            tp = dkeep.sum().item()
            self.detection_tp += tp
            self.detection_fp += nest - tp

            self.total_distance += avg_distance
            self.match_count += 1
            if not torch.isnan(avg_keep_distance):
                self.total_distance_keep += avg_keep_distance
                self.match_count_keep += 1

            # update star/galaxy classification TP/FP
            batch_tgbool = tgbool[b][mtrue][dkeep].reshape(-1)
            batch_egbool = egbool[b][mest][dkeep].reshape(-1)
            self._update_confusion_matrix(batch_tgbool, batch_egbool)

        return match_true, match_est

    def _update_confusion_matrix(self, tgbool: Tensor, egbool: Tensor) -> None:
        """Compute galaxy detection confusion matrix and update TP/FP/TN/FN.

        Args:
            tgbool (Tensor): Galaxy bools from true catalog.
            egbool (Tensor): Galaxy bools from estimated catalog.
        """
        conf_matrix = confusion_matrix(tgbool.cpu(), egbool.cpu(), labels=[1, 0])
        # decompose confusion matrix
        self.gal_tp += conf_matrix[0][0]
        self.gal_fp += conf_matrix[0][1]
        self.star_fp += conf_matrix[1][0]
        self.star_tp += conf_matrix[1][1]

    def _update_classification_metrics(
        self, true: Catalog, est: Catalog, match_true: List, match_est: List
    ) -> None:
        """Update classification metrics based on estimated params at locations of true sources."""
        # only compute classification metrics if available
        if "galaxy_params" not in true or "galaxy_params" not in est:
            return

        # get parameters depending on the kind of catalog
        if self.mode is MetricsMode.FULL:
            params = self._get_classification_params_full(true, est, match_true, match_est)
        elif self.mode is MetricsMode.TILE:
            params = self._get_classification_params_tile(true, est)
        true_gal_params, est_gal_params, true_star_fluxes, est_star_fluxes = params

        # fluxes
        self.star_flux.append(torch.abs(true_star_fluxes - est_star_fluxes).squeeze())

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

    def _get_classification_params_full(self, true, est, match_true, match_est):
        """Get galaxy params in true and est catalogs based on matches."""
        batch_size, max_true, max_est = true.batch_size, true.plocs.shape[1], est.plocs.shape[1]
        true_mask = torch.zeros((batch_size, max_true)).bool()
        est_mask = torch.zeros((batch_size, max_est)).bool()

        for i in range(batch_size):
            true_mask[i][match_true[i]] = True
            est_mask[i][match_est[i]] = True

        true_galaxy_params = true["galaxy_params"][true_mask]
        est_galaxy_params = est["galaxy_params"][est_mask]
        true_star_fluxes = true["star_fluxes"][true_mask]
        est_star_fluxes = est["star_fluxes"][est_mask]

        return true_galaxy_params, est_galaxy_params, true_star_fluxes, est_star_fluxes

    def _get_classification_params_tile(self, true, est):
        """Get galaxy params in est catalog at tiles containing a galaxy in the true catalog."""
        true_indices, true_is_on = true.get_indices_of_on_sources()

        # construct masks for true source type
        true_st = true.gather_param_at_tiles("source_type", true_indices)
        true_gal_bools = true_st == SourceType.GALAXY
        gal_mask = true_gal_bools.squeeze() * true_is_on
        star_mask = (~true_gal_bools).squeeze() * true_is_on

        # gather parameters where source is present in true catalog AND source is actually a star
        # or galaxy respectively in true catalog
        # note that the boolean indexing collapses across batches to return a 1D tensor
        true_gal_params = true.gather_param_at_tiles("galaxy_params", true_indices)[gal_mask]
        est_gal_params = est.gather_param_at_tiles("galaxy_params", true_indices)[gal_mask]
        true_star_fluxes = true.gather_param_at_tiles("star_fluxes", true_indices)[star_mask]
        est_star_fluxes = est.gather_param_at_tiles("star_fluxes", true_indices)[star_mask]

        return true_gal_params, est_gal_params, true_star_fluxes, est_star_fluxes

    def compute(self) -> Dict[str, float]:
        precision = self.detection_tp / (self.detection_tp + self.detection_fp)
        recall = self.detection_tp / self.total_true_n_sources
        f1 = 2 * precision * recall / (precision + recall)

        avg_distance = self.total_distance / self.match_count.item()
        avg_keep_distance = self.total_distance_keep / self.match_count_keep.item()

        class_acc = (self.gal_tp + self.star_tp) / (
            self.gal_tp + self.star_tp + self.gal_fp + self.star_fp
        )

        metrics = {
            "detection_precision": precision.item(),
            "detection_recall": recall.item(),
            "f1": f1.item(),
            "avg_distance": avg_distance.item(),
            "avg_keep_distance": avg_keep_distance.item(),
            "gal_tp": self.gal_tp.item(),
            "gal_fp": self.gal_fp.item(),
            "star_tp": self.star_tp.item(),
            "star_fp": self.star_fp.item(),
            "class_acc": class_acc.item(),
        }

        # add classification metrics if computed
        if self.disk_flux:
            for metric in self.classification_metrics:
                # flatten list of variable-size tensors and take the median
                vals: List = sum([t.flatten().tolist() for t in getattr(self, metric)], [])
                metrics[f"{metric}_mae"] = np.median(vals) if vals else np.nan

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
