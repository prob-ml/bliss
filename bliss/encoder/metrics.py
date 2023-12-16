import torch
from einops import rearrange
from scipy.optimize import linear_sum_assignment
from torchmetrics import Metric

from bliss.catalog import FullCatalog
from bliss.utils.flux_units import convert_nmgy_to_mag


class CatalogMetrics(Metric):
    """Calculates detection and classification metrics between two catalogs.
    Note that galaxy classification metrics are only computed when the values are available in
    both catalogs.
    """

    GALSIM_NAMES = ["disk_frac", "beta_radians", "disk_q", "a_d", "bulge_q", "a_b"]

    def __init__(
        self,
        survey_bands: list,
        dist_slack: float = 1.0,
        mag_slack: float = None,
        mag_slack_band: int = 2,
        mag_bin_cutoffs: list = None,
        dist_sync_on_step: bool = False,
        disable_bar: bool = True,
    ):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.survey_bands = survey_bands
        self.dist_slack = dist_slack
        self.mag_slack = mag_slack
        self.mag_slack_band = mag_slack_band
        self.disable_bar = disable_bar

        cutoffs = mag_bin_cutoffs if mag_bin_cutoffs else []
        self.mag_bin_cutoffs = torch.tensor(cutoffs, device=self.device)

        detection_metrics = [
            "detection_tp",
            "detection_fp",
            "total_true_n_sources",
            "total_match_distance",
            "gal_tp",
            "star_tp",
        ]
        for metric in detection_metrics:
            num_bins = self.mag_bin_cutoffs.shape[0] + 1  # fencepost
            init_val = torch.zeros(num_bins)
            self.add_state(metric, default=init_val, dist_reduce_fx="sum")

        fe_init = torch.zeros(len(self.survey_bands))
        self.add_state("flux_err", default=fe_init, dist_reduce_fx="sum")

        gpe_init = torch.zeros(len(self.GALSIM_NAMES))
        self.add_state("galsim_param_err", default=gpe_init, dist_reduce_fx="sum")

    def update(self, true_cat, est_cat):
        assert isinstance(true_cat, FullCatalog) and isinstance(est_cat, FullCatalog)
        assert true_cat.batch_size == est_cat.batch_size

        if self.mag_slack:
            true_mags = convert_nmgy_to_mag(true_cat.get_fluxes()[:, :, self.mag_slack_band])
            est_mags = convert_nmgy_to_mag(est_cat.get_fluxes()[:, :, self.mag_slack_band])

        for i in range(true_cat.batch_size):
            n_true = int(true_cat.n_sources[i].int().sum().item())
            n_est = int(est_cat.n_sources[i].int().sum().item())
            self.total_true_n_sources += n_true

            tcat_matches, ecat_matches = self.match_catalogs(
                true_cat.plocs[i, 0:n_true],
                est_cat.plocs[i, 0:n_est],
                true_mags[i, 0:n_true] if self.mag_slack else None,
                est_mags[i, 0:n_est] if self.mag_slack else None,
            )

            # update detection stats
            self.detection_tp += len(tcat_matches)
            self.detection_fp += n_est - len(tcat_matches)

            # update match distance
            loc_diffs = true_cat.plocs[i, tcat_matches] - est_cat.plocs[i, ecat_matches]
            image_match_distance = loc_diffs.norm(dim=1).sum()
            self.total_match_distance += image_match_distance

            # update star/galaxy classification stats
            true_gal = true_cat.galaxy_bools[i][tcat_matches]
            est_gal = est_cat.galaxy_bools[i][ecat_matches]
            self.gal_tp += (true_gal * est_gal).sum()
            self.star_tp += (~true_gal * ~est_gal).sum()

            # update total per-band flux errors
            true_flux = true_cat.get_fluxes()[i, tcat_matches, :]
            est_flux = est_cat.get_fluxes()[i, ecat_matches, :]
            self.flux_err += (true_flux - est_flux).abs().sum(dim=0)

            # some real catalogs (e.g., sdss) do not have galaxy_params
            if "galaxy_params" not in true_cat or "galaxy_params" not in est_cat:
                continue

            true_gp = true_cat["galaxy_params"][i, tcat_matches, :]
            est_gp = est_cat["galaxy_params"][i, ecat_matches, :]
            gp_err = (true_gp - est_gp).abs().sum(dim=0)
            # TODO: angle is a special case, need to wrap around pi (not 2pi due to symmetry)
            self.galsim_param_err += gp_err

    def compute(self):
        precision = self.detection_tp / (self.detection_tp + self.detection_fp)
        recall = self.detection_tp / self.total_true_n_sources
        f1 = 2 * precision * recall / (precision + recall)
        avg_match_distance = self.total_match_distance / self.detection_tp
        classification_acc = (self.gal_tp + self.star_tp) / self.detection_tp

        metrics = {
            "detection_precision": precision.item(),
            "detection_recall": recall.item(),
            "detection_f1": f1.item(),
            "avg_match_distance": avg_match_distance.item(),
            "classification_acc": classification_acc.item(),
        }

        avg_flux_err = self.flux_err / self.detection_tp
        for i, band in enumerate(self.survey_bands):
            metrics[f"flux_err_{band}_mae"] = avg_flux_err[i].item()

        if self.gal_tp > 0:
            avg_galsim_param_err = self.galsim_param_err / self.gal_tp
            for i, gs_name in enumerate(self.GALSIM_NAMES):
                metrics[f"{gs_name}_mae"] = avg_galsim_param_err[i]

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

        # Penalize all pairs which are greater than slack apart to favor valid matches
        oob = locs_dist > self.dist_slack

        if self.mag_slack:
            mag_diff = rearrange(true_mags, "i -> i 1") - rearrange(est_mags, "i -> 1 i")
            mag_err = mag_diff.abs()
            oob |= mag_err > self.mag_slack

        cost = locs_dist + oob * 1e20

        # find minimal permutation (can be slow for large catalogs, consider using a heuristic
        # that exploits sparsity instead)
        row_indx, col_indx = linear_sum_assignment(cost.detach().cpu())

        # good match condition: not out-of-bounds due to either slack contraint
        valid_matches = ~oob[row_indx, col_indx].cpu().numpy()

        true_matches = row_indx[valid_matches]
        est_matches = col_indx[valid_matches]

        return true_matches, est_matches
