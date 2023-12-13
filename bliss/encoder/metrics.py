from typing import Dict, List, Union

import torch
from einops import rearrange
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import confusion_matrix
from torch import Tensor
from torchmetrics import Metric

from bliss.catalog import FullCatalog, SourceType, TileCatalog
from bliss.utils.flux_units import convert_nmgy_to_mag

# define type alias to simplify signatures
Catalog = Union[TileCatalog, FullCatalog]


class CatalogMetrics(Metric):
    """Calculates detection and classification metrics between two catalogs.

    CatalogMetrics supports two modes, "matching" and "conditional", which operate on FullCatalog
    objects and TileCatalog objects, respectively.
    For "matching", all metrics are computed by matching predicted sources to true sources.
    For "conditional", detection metrics are still computed that way, but star/galaxy parameter
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
    total_distance_keep: Tensor
    match_count: Tensor
    match_count_keep: Tensor
    gal_tp: Tensor
    gal_fp: Tensor
    star_tp: Tensor
    star_fp: Tensor

    # Classification metrics
    gal_fluxes: Tensor
    star_fluxes: Tensor
    disk_frac: Tensor
    bulge_frac: Tensor
    disk_q: Tensor
    bulge_q: Tensor
    disk_hlr: Tensor
    bulge_hlr: Tensor
    beta_radians: Tensor

    full_state_update: bool = False

    def __init__(
        self,
        survey_bands: list,
        mode: str = "matching",
        dist_slack: float = 1.0,
        mag_slack: float = 0.5,
        mag_slack_band: int = 2,
        dist_sync_on_step: bool = False,
        disable_bar: bool = True,
    ) -> None:
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.survey_bands = survey_bands
        self.mode = mode
        self.dist_slack = dist_slack
        self.mag_slack = mag_slack
        self.mag_slack_band = mag_slack_band
        self.disable_bar = disable_bar

        self.detection_metrics = [
            "detection_tp",
            "detection_fp",
            "total_true_n_sources",
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
            num_zeros = len(self.survey_bands) if metric in {"gal_fluxes", "star_fluxes"} else 1
            default = torch.zeros(num_zeros)
            self.add_state(metric, default=default, dist_reduce_fx="sum")

    def update(self, true: Catalog, est: Catalog) -> None:
        assert true.batch_size == est.batch_size
        if self.mode == "matching":
            msg = "Metrics mode is set to `matching` but received a TileCatalog"
            assert isinstance(true, FullCatalog) and isinstance(est, FullCatalog), msg
        elif self.mode == "conditional":
            msg = "Metrics mode is set to `conditional` but received a FullCatalog"
            assert isinstance(true, TileCatalog) and isinstance(est, TileCatalog), msg

        match_true, match_est = self._update_detection_metrics(true, est)
        self._update_classification_metrics(true, est, match_true, match_est)

    def _update_detection_metrics(self, true: Catalog, est: Catalog) -> None:
        """Update detection metrics."""
        if self.mode == "matching":
            true_locs = true.plocs
            est_locs = est.plocs
            tgbool = true.galaxy_bools
            egbool = est.galaxy_bools
            true_fluxes = true.get_fluxes_of_on_sources()[:, :, self.mag_slack_band]
            est_fluxes = est.get_fluxes_of_on_sources()[:, :, self.mag_slack_band]
            true_mags = convert_nmgy_to_mag(true_fluxes)
            est_mags = convert_nmgy_to_mag(est_fluxes)
        elif self.mode == "conditional":
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

            true_mags = torch.zeros_like(true_on_idx, dtype=torch.float)
            est_mags = true_mags  # hack to avoid filtering on mag in conditional mode

        match_true, match_est = [], []
        for i in range(true.batch_size):
            ntrue = int(true.n_sources[i].int().sum().item())
            nest = int(est.n_sources[i].int().sum().item())
            self.total_true_n_sources += ntrue

            # if either ntrue or nest are 0, manually increment FP/FN and continue
            if ntrue == 0 or nest == 0:
                if nest > 0:
                    self.detection_fp += nest
                match_true.append([])
                match_est.append([])
                continue

            mtrue, mest, dkeep, avg_keep_distance = self.match_catalogs(
                true_locs[i, 0:ntrue],
                est_locs[i, 0:nest],
                true_mags[i, 0:ntrue],
                est_mags[i, 0:nest],
            )
            match_true.append(mtrue[dkeep])
            match_est.append(mest[dkeep])

            # update TP/FP and distances
            tp = dkeep.sum().item()
            self.detection_tp += tp
            self.detection_fp += nest - tp

            self.match_count += 1
            if not torch.isnan(avg_keep_distance):
                self.total_distance_keep += avg_keep_distance
                self.match_count_keep += 1

            # update star/galaxy classification TP/FP
            batch_tgbool = tgbool[i][mtrue][dkeep].reshape(-1)
            batch_egbool = egbool[i][mest][dkeep].reshape(-1)
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
        if self.mode == "matching":
            if not (match_true and match_est):
                return  # need matches to compute classification metrics on full catalog
            true_params, est_params = self._get_classification_params_full(
                true, est, match_true, match_est
            )
        elif self.mode == "conditional":
            true_params, est_params = self._get_classification_params_tile(true, est)

        true_gal_fluxes = true_params["galaxy_fluxes"]
        true_star_fluxes = true_params["star_fluxes"]
        true_gal_params = true_params["galaxy_params"]
        est_gal_fluxes = est_params["galaxy_fluxes"]
        est_star_fluxes = est_params["star_fluxes"]
        est_gal_params = est_params["galaxy_params"]

        self.gal_fluxes = (true_gal_fluxes - est_gal_fluxes).abs().mean(dim=0)
        self.star_fluxes = (true_star_fluxes - est_star_fluxes).abs().mean(dim=0)

        # skip if no galaxies in true or estimated catalog
        if (true_gal_params.shape[0] == 0) or (est_gal_params.shape[0] == 0):
            return

        # disk/bulge proportions
        self.disk_frac = (true_gal_params[:, 0] - est_gal_params[:, 0]).abs().mean()
        self.bulge_frac = ((1 - true_gal_params[:, 0]) - (1 - est_gal_params[:, 0])).abs().mean()

        # angle
        self.beta_radians = (true_gal_params[:, 1] - est_gal_params[:, 1]).abs().mean()

        # axis ratio
        self.disk_q = (true_gal_params[:, 2] - est_gal_params[:, 2]).abs().mean()
        self.bulge_q = (true_gal_params[:, 4] - est_gal_params[:, 4]).abs().mean()

        # half-light radius
        # sqrt(a * b) = sqrt(a * a * q) = a * sqrt(q)
        est_disk_hlr = est_gal_params[:, 3] * torch.sqrt(est_gal_params[:, 2])
        true_disk_hlr = true_gal_params[:, 3] * torch.sqrt(true_gal_params[:, 2])
        self.disk_hlr = (true_disk_hlr - est_disk_hlr).abs().mean()

        est_bulge_hlr = est_gal_params[:, 5] * torch.sqrt(est_gal_params[:, 4])
        true_bulge_hlr = true_gal_params[:, 5] * torch.sqrt(true_gal_params[:, 4])
        self.bulge_hlr = (true_bulge_hlr - est_bulge_hlr).abs().mean()

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
        gal_mask = true_gal_bools.squeeze() * true_is_on.squeeze(1)
        star_mask = (~true_gal_bools).squeeze() * true_is_on.squeeze(1)

        # gather parameters where source is present in true catalog AND source is actually a star
        # or galaxy respectively in true catalog
        # note that the boolean indexing collapses across batches to return a 1D tensor
        param_masks = {
            "galaxy_fluxes": gal_mask,
            "star_fluxes": star_mask,
            "galaxy_params": gal_mask,
        }
        true_params = {}
        est_params = {}
        for param, mask in param_masks.items():
            true_params[param] = true.gather_param_at_tiles(param, true_indices).squeeze(1)[mask]
            est_params[param] = est.gather_param_at_tiles(param, true_indices).squeeze(1)[mask]

        return true_params, est_params

    def compute(self) -> Dict[str, float]:
        precision = self.detection_tp / (self.detection_tp + self.detection_fp)
        recall = self.detection_tp / self.total_true_n_sources
        f1 = 2 * precision * recall / (precision + recall)

        avg_keep_distance = self.total_distance_keep / self.match_count_keep.item()

        class_acc = (self.gal_tp + self.star_tp) / (
            self.gal_tp + self.star_tp + self.gal_fp + self.star_fp
        )

        metrics = {
            "detection_precision": precision.item(),
            "detection_recall": recall.item(),
            "f1": f1.item(),
            "avg_keep_distance": avg_keep_distance.item(),
            "gal_tp": self.gal_tp.item(),
            "gal_fp": self.gal_fp.item(),
            "star_tp": self.star_tp.item(),
            "star_fp": self.star_fp.item(),
            "class_acc": class_acc.item(),
        }

        # add classification metrics if computed
        for metric in self.classification_metrics:
            v = getattr(self, metric, None)

            # take median along first dim of stacked tensors, i.e. across images
            if metric in {"gal_fluxes", "star_fluxes"}:
                for i, band in enumerate(self.survey_bands):
                    metrics[f"{metric}_{band}_mae"] = v[i].item()
            else:
                metrics[f"{metric}_mae"] = v.item()

        return metrics

    def match_catalogs(self, true_locs, est_locs, true_mags, est_mags):
        """Match true and estimated locations and returned indices to match.

        Permutes `est_locs` to find minimal error between `true_locs` and `est_locs`.
        The matching is done with the Hungarian algorithm.

        Args:
            true_locs: Tensor of shape `(n1 x 2)`, where `n1` is the true number of sources.
                The centroids should be in units of pixels.
            est_locs: Tensor of shape `(n2 x 2)`, where `n2` is the predicted
                number of sources. The centroids should be in units of pixels.
            true_mags: The true magnitudes of the sources in a particular band
            est_mags: The estimated magnitudes of the sources in a particular band

        Returns:
            A tuple of the following objects:
            - row_indx: Indices of true objects matched to estimated objects.
            - col_indx: Indices of estimated objects matched to true objects.
            - dist_keep: Matched objects to keep based on L2 distances.
            - avg_keep_distance: Average L2 distance over matched objects to keep.
        """
        locs_diff = rearrange(true_locs, "i j -> i 1 j") - rearrange(est_locs, "i j -> 1 i j")
        locs_dist = locs_diff.norm(dim=2)

        mag_diff = rearrange(true_mags, "i -> i 1") - rearrange(est_mags, "i -> 1 i")
        mag_err = mag_diff.abs()

        # Penalize all pairs which are greater than slack apart to favor valid matches
        oob = (locs_dist > self.dist_slack) | (mag_err > self.mag_slack)
        cost = locs_dist + oob * 1e20

        # find minimal permutation
        row_indx, col_indx = linear_sum_assignment(cost.detach().cpu())

        # good match condition: not out-of-bounds due to either slack contraint
        matches_to_keep = ~oob[row_indx, col_indx]
        match_dist = (true_locs[row_indx] - est_locs[col_indx]).norm(dim=1)
        avg_keep_distance = match_dist[matches_to_keep].mean()

        return row_indx, col_indx, matches_to_keep.cpu().numpy(), avg_keep_distance
