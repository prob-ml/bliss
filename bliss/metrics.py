# pylint: disable=E1101
# mypy: disable-error-code="union-attr"

from enum import Enum
from typing import Dict, List, Union

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

    # Classification metrics
    gal_fluxes: List
    star_fluxes: List
    disk_frac: List
    bulge_frac: List
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
            "disk_frac",
            "bulge_frac",
            "disk_q",
            "bulge_q",
            "disk_hlr",
            "bulge_hlr",
            "beta_radians",
            "gal_fluxes",
            "star_fluxes",
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
            true_params, est_params = self._get_classification_params_full(
                true, est, match_true, match_est
            )
        elif self.mode is MetricsMode.TILE:
            true_params, est_params = self._get_classification_params_tile(true, est)

        true_gal_fluxes = true_params["galaxy_fluxes"]
        true_star_fluxes = true_params["star_fluxes"]
        true_gal_params = true_params["galaxy_params"]
        est_gal_fluxes = est_params["galaxy_fluxes"]
        est_star_fluxes = est_params["star_fluxes"]
        est_gal_params = est_params["galaxy_params"]

        # fluxes
        self.gal_fluxes.append(torch.abs(true_gal_fluxes - est_gal_fluxes))
        self.star_fluxes.append(torch.abs(true_star_fluxes - est_star_fluxes))

        # disk/bulge proportions
        self.disk_frac.append(torch.abs(true_gal_params[:, 0] - est_gal_params[:, 0]))
        self.bulge_frac.append(torch.abs((1 - true_gal_params[:, 0]) - (1 - est_gal_params[:, 0])))

        # angle
        self.beta_radians.append(torch.abs(true_gal_params[:, 1] - est_gal_params[:, 1]))

        # axis ratio
        self.disk_q.append(torch.abs(true_gal_params[:, 2] - est_gal_params[:, 2]))
        self.bulge_q.append(torch.abs(true_gal_params[:, 4] - est_gal_params[:, 4]))

        # half-light radius
        # sqrt(a * b) = sqrt(a * a * q) = a * sqrt(q)
        est_disk_hlr = est_gal_params[:, 3] * torch.sqrt(est_gal_params[:, 2])
        true_disk_hlr = true_gal_params[:, 3] * torch.sqrt(true_gal_params[:, 2])
        self.disk_hlr.append(torch.abs(true_disk_hlr - est_disk_hlr))

        est_bulge_hlr = est_gal_params[:, 5] * torch.sqrt(est_gal_params[:, 4])
        true_bulge_hlr = true_gal_params[:, 5] * torch.sqrt(true_gal_params[:, 4])
        self.bulge_hlr.append(torch.abs(true_bulge_hlr - est_bulge_hlr))

    def _get_classification_params_full(self, true, est, match_true, match_est):
        """Get galaxy params in true and est catalogs based on matches."""
        batch_size, max_true, max_est = true.batch_size, true.plocs.shape[1], est.plocs.shape[1]
        true_mask = torch.zeros((batch_size, max_true)).bool()
        est_mask = torch.zeros((batch_size, max_est)).bool()

        for i in range(batch_size):
            true_mask[i][match_true[i]] = True
            est_mask[i][match_est[i]] = True

        params = ["galaxy_fluxes", "star_fluxes", "galaxy_params"]
        true_params = {param: true[param][true_mask] for param in params}
        est_params = {param: est[param][est_mask] for param in params}

        return true_params, est_params

    def _get_classification_params_tile(self, true, est):
        """Get galaxy params in est catalog at tiles containing a galaxy in the true catalog."""
        true_indices, true_is_on = true.get_indices_of_on_sources()

        # construct flattened masks for true source type
        true_st = true.gather_param_at_tiles("source_type", true_indices)
        true_gal_bools = true_st == SourceType.GALAXY
        gal_mask = true_gal_bools.squeeze() * true_is_on
        star_mask = (~true_gal_bools).squeeze() * true_is_on

        # gather parameters where source is present in true catalog AND source is actually a star
        # or galaxy respectively in true catalog
        # note that the boolean indexing collapses across batches to return a 1D tensor
        params = {"galaxy_fluxes": gal_mask, "star_fluxes": star_mask, "galaxy_params": gal_mask}
        true_params = {
            param: true.gather_param_at_tiles(param, true_indices)[mask]
            for param, mask in params.items()
        }
        est_params = {
            param: est.gather_param_at_tiles(param, true_indices)[mask]
            for param, mask in params.items()
        }

        return true_params, est_params

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
        for metric in self.classification_metrics:
            val_list = getattr(self, metric, None)
            len_val = len(val_list[0])
            if (len_val == 0) or (not val_list):
                continue

            # take median along first dim of stacked tensors, i.e. across images
            median_vals = torch.cat(val_list, dim=0).median(dim=0).values

            if metric in {"gal_fluxes", "star_fluxes"}:
                metrics.update(
                    {
                        f"{metric}_{band}_mae": median_vals[i].item()
                        for i, band in enumerate("ugriz")
                    }
                )
            else:
                metrics[f"{metric}_mae"] = median_vals.item()

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

    return row_indx, col_indx, dist_keep.cpu().numpy(), avg_distance, avg_keep_distance


def three_way_matching(pred_cat, comp_cat, gt_cat, slack=1):
    """Performs a 3-way matching between two catalogs and a ground truth catalog.

    Args:
        pred_cat: predicted catalog
        comp_cat: catalog to compare to
        gt_cat: catalog to use as "ground truth"
        slack: l-infinity threshold for matching objects

    Returns:
        Dict: a dictionary of matches between sets of catalogs.
            gt_all: all gt sources that matched either pred or comp
            gt_pred_only: gt sources that matched pred but not comp
            gt_comp_only: gt sources that matched comp but not pred
            pred_only: pred sources that did not match gt
            comp_only: comp sources that did not match gt
    """
    gt_locs, pred_locs, comp_locs = gt_cat.plocs[0], pred_cat.plocs[0], comp_cat.plocs[0]
    # compute matches for both catalogs against gt
    match_gt_pred = match_by_locs(gt_locs, pred_locs, slack=slack)
    match_gt_comp = match_by_locs(gt_locs, comp_locs, slack=slack)

    gt_pred_matches = match_gt_pred[0][match_gt_pred[2]]  # get indices to keep based on distance
    pred_gt_matches = match_gt_pred[1][match_gt_pred[2]]
    gt_comp_matches = match_gt_comp[0][match_gt_comp[2]]
    comp_gt_matches = match_gt_comp[1][match_gt_comp[2]]

    return {
        # in gt and pred or comp
        "gt_all": set(match_gt_pred[0]).union(match_gt_comp[0]),
        # in pred and gt, not in comp
        "gt_pred_only": set(gt_pred_matches).difference(gt_comp_matches),
        # in comp and gt, not in pred
        "gt_comp_only": set(gt_comp_matches).difference(gt_pred_matches),
        # in pred, not in gt
        "pred_only": set(range(pred_cat.n_sources.item())).difference(pred_gt_matches),
        # in comp, not in gt
        "comp_only": set(range(comp_cat.n_sources.item())).difference(comp_gt_matches),
    }
