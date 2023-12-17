import torch
from einops import rearrange
from scipy.optimize import linear_sum_assignment
from torchmetrics import Metric

from bliss.catalog import FullCatalog


class CatalogMatcher:
    def __init__(
        self,
        dist_slack: float = 1.0,
        mag_slack: float = None,
        mag_band: int = 2,
    ):
        self.dist_slack = dist_slack
        self.mag_slack = mag_slack
        self.mag_band = mag_band

    def match_catalogs(self, true_cat, est_cat):
        assert isinstance(true_cat, FullCatalog) and isinstance(est_cat, FullCatalog)
        assert true_cat.batch_size == est_cat.batch_size

        if self.mag_slack:
            true_mags = true_cat.magnitudes[:, :, self.mag_band]
            est_mags = est_cat.magnitudes[:, :, self.mag_band]

        matching = []
        for i in range(true_cat.batch_size):
            n_true = int(true_cat.n_sources[i].int().sum().item())
            n_est = int(est_cat.n_sources[i].int().sum().item())

            true_locs = true_cat.plocs[i, :n_true]
            est_locs = est_cat.plocs[i, :n_est]

            locs_diff = rearrange(true_locs, "i j -> i 1 j") - rearrange(est_locs, "i j -> 1 i j")
            locs_dist = locs_diff.norm(dim=2)

            # Penalize all pairs which are greater than slack apart to favor valid matches
            oob = locs_dist > self.dist_slack

            if self.mag_slack:
                true_mag_col = rearrange(true_mags[i, :n_true], "k -> k 1")
                est_mag_col = rearrange(est_mags[i, :n_est], "k -> 1 k")
                mag_err = (true_mag_col - est_mag_col).abs()
                oob |= mag_err > self.mag_slack

            cost = locs_dist + oob * 1e20

            # find minimal permutation (can be slow for large catalogs, consider using a heuristic
            # that exploits sparsity instead)
            row_indx, col_indx = linear_sum_assignment(cost.detach().cpu())

            # good match condition: not out-of-bounds due to either slack contraint
            valid_matches = ~oob[row_indx, col_indx].cpu().numpy()

            true_matches = torch.from_numpy(row_indx[valid_matches])
            est_matches = torch.from_numpy(col_indx[valid_matches])
            matching.append((true_matches, est_matches))

        return matching


class DetectionPerformance(Metric):
    """Calculates detection and classification metrics between two catalogs.
    Note that galaxy classification metrics are only computed when the values are available in
    both catalogs.
    """

    def __init__(
        self,
        mag_band: int = 2,
        mag_bin_cutoffs: list = None,
    ):
        super().__init__()

        self.mag_band = mag_band
        self.mag_bin_cutoffs = mag_bin_cutoffs if mag_bin_cutoffs else []

        detection_metrics = [
            "n_true_sources",
            "n_est_sources",
            "n_true_matches",
            "n_est_matches",
        ]
        for metric in detection_metrics:
            n_bins = len(self.mag_bin_cutoffs) + 1  # fencepost
            init_val = torch.zeros(n_bins)
            self.add_state(metric, default=init_val, dist_reduce_fx="sum")

    def update(self, true_cat, est_cat, matching):
        if self.mag_band:
            true_mags = true_cat.magnitudes[:, :, self.mag_band].contiguous()
            est_mags = est_cat.magnitudes[:, :, self.mag_band].contiguous()
        else:
            # hack to match regardless of magnitude; intended for
            # catalogs from surveys with incompatible filter bands
            true_mags = torch.ones_like(true_cat.plocs[:, :, 0])
            est_mags = torch.ones_like(est_cat.plocs[:, :, 0])

        for i in range(true_cat.batch_size):
            tcat_matches, ecat_matches = matching[i]
            n_true = true_cat.n_sources[i].sum().item()
            n_est = est_cat.n_sources[i].sum().item()

            cutoffs = torch.tensor(self.mag_bin_cutoffs, device=self.device)
            n_bins = len(cutoffs) + 1

            tmi = true_mags[i, 0:n_true]
            emi = est_mags[i, 0:n_est]
            tmim, emim = tmi[tcat_matches], emi[ecat_matches]

            self.n_true_sources += torch.bucketize(tmi, cutoffs).bincount(minlength=n_bins)
            self.n_est_sources += torch.bucketize(emi, cutoffs).bincount(minlength=n_bins)
            self.n_true_matches += torch.bucketize(tmim, cutoffs).bincount(minlength=n_bins)
            self.n_est_matches += torch.bucketize(emim, cutoffs).bincount(minlength=n_bins)

    def compute(self):
        precision = self.n_est_matches.sum() / self.n_est_sources.sum()
        recall = self.n_true_matches.sum() / self.n_true_sources.sum()
        f1 = 2 * precision * recall / (precision + recall)

        return {
            "detection_precision": precision,
            "detection_recall": recall,
            "detection_f1": f1,
        }


class SourceTypeAccuracy(Metric):
    def __init__(self):
        super().__init__()

        self.add_state("gal_tp", default=torch.zeros(1), dist_reduce_fx="sum")
        self.add_state("star_tp", default=torch.zeros(1), dist_reduce_fx="sum")
        self.add_state("n_matches", default=torch.zeros(1), dist_reduce_fx="sum")

    def update(self, true_cat, est_cat, matching):
        for i in range(true_cat.batch_size):
            tcat_matches, ecat_matches = matching[i]
            self.n_matches += tcat_matches.size(0)

            true_gal = true_cat.galaxy_bools[i][tcat_matches]
            est_gal = est_cat.galaxy_bools[i][ecat_matches]

            self.gal_tp += (true_gal * est_gal).sum()
            self.star_tp += (~true_gal * ~est_gal).sum()

    def compute(self):
        acc = (self.gal_tp + self.star_tp) / self.n_matches
        return {"classification_acc": acc.item()}


class FluxError(Metric):
    def __init__(self, survey_bands):
        super().__init__()
        self.survey_bands = survey_bands

        fe_init = torch.zeros(len(self.survey_bands))
        self.add_state("flux_err", default=fe_init, dist_reduce_fx="sum")

        self.add_state("n_matches", default=torch.zeros(1), dist_reduce_fx="sum")

    def update(self, true_cat, est_cat, matching):
        for i in range(true_cat.batch_size):
            tcat_matches, ecat_matches = matching[i]
            self.n_matches += tcat_matches.size(0)

            true_flux = true_cat.on_fluxes[i, tcat_matches, :]
            est_flux = est_cat.on_fluxes[i, ecat_matches, :]
            self.flux_err += (true_flux - est_flux).abs().sum(dim=0)

    def compute(self):
        avg_flux_err = self.flux_err / self.n_matches
        results = {}
        for i, band in enumerate(self.survey_bands):
            results[f"flux_err_{band}_mae"] = avg_flux_err[i].item()
        return results


class GalaxyShapeError(Metric):
    GALSIM_NAMES = ["disk_frac", "beta_radians", "disk_q", "a_d", "bulge_q", "a_b"]

    def __init__(self):
        super().__init__()

        gpe_init = torch.zeros(len(self.GALSIM_NAMES))
        self.add_state("galsim_param_err", default=gpe_init, dist_reduce_fx="sum")

        self.add_state("n_true_galaxies", default=torch.zeros(1), dist_reduce_fx="sum")

    def update(self, true_cat, est_cat, matching):
        for i in range(true_cat.batch_size):
            tcat_matches, ecat_matches = matching[i]

            true_gal = true_cat.galaxy_bools[i][tcat_matches]
            self.n_true_galaxies += true_gal.sum()

            # TODO: only compute error for *true* galaxies
            true_gp = true_cat["galaxy_params"][i, tcat_matches, :]
            est_gp = est_cat["galaxy_params"][i, ecat_matches, :]
            gp_err = (true_gp - est_gp).abs().sum(dim=0)
            # TODO: angle is a special case, need to wrap around pi (not 2pi due to symmetry)
            self.galsim_param_err += gp_err

    def compute(self):
        avg_galsim_param_err = self.galsim_param_err / self.n_true_galaxies
        results = {}
        for i, gs_name in enumerate(self.GALSIM_NAMES):
            results[f"{gs_name}_mae"] = avg_galsim_param_err[i]

        return results
