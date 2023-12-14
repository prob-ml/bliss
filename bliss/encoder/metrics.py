import torch
from einops import rearrange
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import confusion_matrix
from torch import Tensor
from torchmetrics import Metric

from bliss.catalog import FullCatalog
from bliss.utils.flux_units import convert_nmgy_to_mag


class CatalogMetrics(Metric):
    """Calculates detection and classification metrics between two catalogs.
    Note that galaxy classification metrics are only computed when the values are available in
    both catalogs.
    """

    # detection metrics
    detection_tp: Tensor
    detection_fp: Tensor
    total_true_n_sources: Tensor
    total_match_distance: Tensor
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
        dist_slack: float = 1.0,
        mag_slack: float = float("inf"),
        mag_slack_band: int = 2,
        mag_bin_cutoffs: list = (),
        dist_sync_on_step: bool = False,
        disable_bar: bool = True,
    ):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.survey_bands = survey_bands
        self.dist_slack = dist_slack
        self.mag_slack = mag_slack
        self.mag_slack_band = mag_slack_band
        self.mag_bin_cutoffs = torch.tensor(mag_bin_cutoffs, device=self.device)
        self.disable_bar = disable_bar

        detection_metrics = [
            "detection_tp",
            "detection_fp",
            "total_true_n_sources",
            "total_match_distance",
            "gal_tp",
            "gal_fp",
            "star_tp",
            "star_fp",
        ]
        for metric in detection_metrics:
            num_bins = self.mag_bin_cutoffs.shape[0] + 1  # fence post
            init_val = torch.zeros(num_bins)
            self.add_state(metric, default=init_val, dist_reduce_fx="sum")

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

    def update(self, true_cat, est_cat):
        assert isinstance(true_cat, FullCatalog) and isinstance(est_cat, FullCatalog)
        assert true_cat.batch_size == est_cat.batch_size
        batch_true_matches, batch_est_matches = self._update_detection_metrics(true_cat, est_cat)
        self._update_classification_metrics(
            true_cat, est_cat, batch_true_matches, batch_est_matches
        )

    def _update_detection_metrics(self, true_cat, est_cat) -> None:
        """Update detection metrics."""
        true_mags = convert_nmgy_to_mag(true_cat.get_fluxes()[:, :, self.mag_slack_band])
        est_mags = convert_nmgy_to_mag(est_cat.get_fluxes()[:, :, self.mag_slack_band])

        batch_true_matches, batch_est_matches = [], []
        for i in range(true_cat.batch_size):
            n_true = int(true_cat.n_sources[i].int().sum().item())
            n_est = int(est_cat.n_sources[i].int().sum().item())
            self.total_true_n_sources += n_true

            true_matches, est_matches = self.match_catalogs(
                true_cat.plocs[i, 0:n_true],
                est_cat.plocs[i, 0:n_est],
                true_mags[i, 0:n_true],
                est_mags[i, 0:n_est],
            )
            batch_true_matches.append(true_matches)
            batch_est_matches.append(est_matches)

            # update TP/FP
            tp = len(true_matches)
            self.detection_tp += tp
            self.detection_fp += n_est - tp

            # update match distance
            loc_diffs = true_cat.plocs[i, true_matches] - est_cat.plocs[i, est_matches]
            image_match_distance = loc_diffs.norm(dim=1).sum()
            self.total_match_distance += image_match_distance

            # update star/galaxy classification TP/FP
            batch_tgbool = true_cat.galaxy_bools[i][true_matches].reshape(-1)
            batch_egbool = est_cat.galaxy_bools[i][est_matches].reshape(-1)

            conf_matrix = confusion_matrix(batch_tgbool.cpu(), batch_egbool.cpu(), labels=[1, 0])

            self.gal_tp += conf_matrix[0][0]
            self.gal_fp += conf_matrix[0][1]
            self.star_fp += conf_matrix[1][0]
            self.star_tp += conf_matrix[1][1]

        return batch_true_matches, batch_est_matches

    def _update_classification_metrics(
        self, true_cat, est_cat, batch_true_matches, batch_est_matches
    ):
        """Update classification metrics based on estimated params at locations of true sources."""
        # only compute classification metrics if available
        if "galaxy_params" not in true_cat or "galaxy_params" not in est_cat:
            return

        # get parameters depending on the kind of catalog
        if not (batch_true_matches and batch_est_matches):
            return  # need matches to compute classification metrics on full catalog

        # Get galaxy params in true and est catalogs based on matches
        batch_size, max_true, max_est = (
            true_cat.batch_size,
            true_cat.plocs.shape[1],
            est_cat.plocs.shape[1],
        )
        true_mask = torch.zeros((batch_size, max_true)).bool()
        est_mask = torch.zeros((batch_size, max_est)).bool()
        for i in range(batch_size):
            true_mask[i][batch_true_matches[i]] = True
            est_mask[i][batch_est_matches[i]] = True
        params = ["galaxy_fluxes", "star_fluxes", "galaxy_params"]
        true_params = {param: true_cat[param][true_mask] for param in params}
        est_params = {param: est_cat[param][est_mask] for param in params}

        # star flux average absolute error
        star_flux_diff = true_params["star_fluxes"] - est_params["star_fluxes"]
        self.star_fluxes = star_flux_diff.abs().mean(dim=0)

        # galaxy cluster average absolute error
        gal_flux_diff = true_params["galaxy_fluxes"] - est_params["galaxy_fluxes"]
        self.gal_fluxes = gal_flux_diff.abs().mean(dim=0)

        # galaxy parameters average absolute error
        true_gal_params = true_params["galaxy_params"]
        est_gal_params = est_params["galaxy_params"]

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

    def compute(self):
        precision = self.detection_tp / (self.detection_tp + self.detection_fp)
        recall = self.detection_tp / self.total_true_n_sources
        f1 = 2 * precision * recall / (precision + recall)

        avg_match_distance = self.total_match_distance / self.detection_tp

        n_predictions = self.gal_tp + self.star_tp + self.gal_fp + self.star_fp
        classification_acc = (self.gal_tp + self.star_tp) / n_predictions

        metrics = {
            "detection_precision": precision.item(),
            "detection_recall": recall.item(),
            "f1": f1.item(),
            "avg_match_distance": avg_match_distance.item(),
            "classification_acc": classification_acc.item(),
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
            - true_matches: Indices of true objects matched to estimated objects.
            - est_matches: Indices of estimated objects matched to true objects.
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
        valid_matches = ~oob[row_indx, col_indx].cpu().numpy()

        true_matches = row_indx[valid_matches]
        est_matches = col_indx[valid_matches]

        return true_matches, est_matches
