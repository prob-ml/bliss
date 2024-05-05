import seaborn as sns
import torch
from einops import rearrange
from matplotlib import pyplot as plt
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
        exclude_last_bin: bool = False,
    ):
        super().__init__()

        self.mag_band = mag_band
        self.mag_bin_cutoffs = mag_bin_cutoffs if mag_bin_cutoffs else []
        self.exclude_last_bin = exclude_last_bin

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
            # handle single band prediction
            est_mag_band = 0 if est_cat["galaxy_fluxes"].shape[-1] == 1 else self.mag_band
            true_mags = true_cat.magnitudes[:, :, self.mag_band].contiguous()
            est_mags = est_cat.magnitudes[:, :, est_mag_band].contiguous()
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
        final_idx = -1 if self.exclude_last_bin else None
        n_est_matches = self.n_est_matches[:final_idx]
        n_true_matches = self.n_true_matches[:final_idx]
        n_est_sources = self.n_est_sources[:final_idx]
        n_true_sources = self.n_true_sources[:final_idx]

        precision = n_est_matches.sum() / n_est_sources.sum()
        recall = n_true_matches.sum() / n_true_sources.sum()
        f1 = 2 * precision * recall / (precision + recall)

        binned_precision = n_est_matches / n_est_sources
        binned_recall = n_true_matches / n_true_sources
        binned_f1 = 2 * binned_precision * binned_recall / (binned_precision + binned_recall)

        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "binned_precision": binned_precision,
            "binned_recall": binned_recall,
            "binned_f1": binned_f1,
        }

    def _plot(self):
        # Compute recall, precision, and F1 score
        recall = self.n_true_matches / self.n_true_sources
        precision = self.n_est_matches / self.n_est_sources
        f1 = 2 * precision * recall / (precision + recall)

        mbc = self.mag_bin_cutoffs
        if not mbc:
            return None, None

        xlabels = [f"[{mbc[i]}, {mbc[i+1]}]" for i in range(len(mbc) - 1)]
        xlabels = ["< " + str(mbc[0])] + xlabels + ["> " + str(mbc[-1])]

        if self.exclude_last_bin:
            precision = precision[:-1]
            recall = recall[:-1]
            f1 = f1[:-1]
            xlabels = xlabels[:-1]

        sns.set(style="whitegrid")
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        axes[0].plot(recall.tolist(), marker="s")
        axes[0].set_title("Recall")
        axes[0].set_xticks(range(len(xlabels)))
        axes[0].set_xticklabels(xlabels, rotation=45)
        axes[0].set_ylim([0, 1])

        axes[1].plot(precision.tolist(), marker="s")
        axes[1].set_title("Precision")
        axes[1].set_xticks(range(len(xlabels)))
        axes[1].set_xticklabels(xlabels, rotation=45)
        axes[1].set_ylim([0, 1])

        axes[2].plot(f1.tolist(), marker="s")
        axes[2].set_title("F1 Score")
        axes[2].set_xticks(range(len(xlabels)))
        axes[2].set_xticklabels(xlabels, rotation=45)
        axes[2].set_ylim([0, 1])

        plt.tight_layout()
        plt.show()

        return fig, axes


class SourceTypeAccuracy(Metric):
    def __init__(self, mag_band=2, mag_bin_cutoffs=[], exclude_last_bin=False,):
        super().__init__()

        self.mag_band = mag_band
        self.mag_bin_cutoffs = mag_bin_cutoffs
        self.n_bins = len(self.mag_bin_cutoffs) + 1
        self.exclude_last_bin = exclude_last_bin

        self.add_state("gal_tp", default=torch.zeros(self.n_bins), dist_reduce_fx="sum")
        self.add_state("star_tp", default=torch.zeros(self.n_bins), dist_reduce_fx="sum")
        self.add_state("n_matches", default=torch.zeros(self.n_bins), dist_reduce_fx="sum")


    def update(self, true_cat, est_cat, matching):
        true_mags = true_cat.magnitudes[:, :, self.mag_band].contiguous()
        boundaries = torch.tensor(self.mag_bin_cutoffs, device=self.device)

        for i in range(true_cat.batch_size):
            tcat_matches, ecat_matches = matching[i]
            n_true = true_cat.n_sources[i].sum().item()

            true_matched_mags = true_mags[i, 0:n_true][tcat_matches]
            bins = torch.bucketize(true_matched_mags, boundaries)

            true_gal = true_cat.galaxy_bools[i][tcat_matches]
            est_gal = est_cat.galaxy_bools[i][ecat_matches]

            gal_tp = torch.zeros(self.n_bins, device=self.device).float()
            gal_tp = gal_tp.scatter_add(0, bins, (true_gal * est_gal).float().squeeze())
            self.gal_tp += gal_tp

            star_tp = torch.zeros(self.n_bins, device=self.device).float()
            star_tp += star_tp.scatter_add(0, bins, (~true_gal * ~est_gal).float().squeeze())
            self.star_tp += star_tp

            self.n_matches += bins.bincount(minlength=self.n_bins)

    def compute(self):
        final_idx = -1 if self.exclude_last_bin else None
        gal_tp = self.gal_tp[:final_idx]
        star_tp = self.star_tp[:final_idx]
        n_matches = self.n_matches[:final_idx]
        binned_acc = (gal_tp + star_tp) / n_matches
        total_acc = (gal_tp.sum() + star_tp.sum()) / n_matches.sum()
        return {
            "class_acc": total_acc,
            "binned_class_acc": binned_acc,
        }


class FluxError(Metric):
    def __init__(
            self,
            bands,
            survey_bands,
            mag_band: int = 2,
            mag_bin_cutoffs: list = [],
            exclude_last_bin: bool = False
        ):
        super().__init__()
        self.bands = torch.tensor(bands)  # list of band indices (e.g. 2)
        self.survey_bands = survey_bands  # list of band names (e.g. "r")

        self.mag_band = mag_band
        self.mag_bin_cutoffs = mag_bin_cutoffs
        self.exclude_last_bin = exclude_last_bin
        self.n_bins = len(self.mag_bin_cutoffs) + 1

        fe_init = torch.zeros((len(self.bands), self.n_bins))  # n_bins per band
        self.add_state("flux_err", default=fe_init, dist_reduce_fx="sum")
        self.add_state("n_matches", default=torch.zeros(self.n_bins), dist_reduce_fx="sum")

    def update(self, true_cat, est_cat, matching):
        if self.mag_band:
            true_mags = true_cat.magnitudes[:, :, self.mag_band].contiguous()
        else:
            true_mags = torch.ones_like(true_cat.plocs[:, :, 0])
        
        boundaries = torch.tensor(self.mag_bin_cutoffs, device=self.device)

        for i in range(true_cat.batch_size):
            tcat_matches, ecat_matches = matching[i]
            n_true = true_cat.n_sources[i].sum().item()
            true_matched_mags = true_mags[i, 0:n_true][tcat_matches]

            bins = torch.bucketize(true_matched_mags, boundaries)

            true_flux = true_cat.on_fluxes[i, tcat_matches, self.bands]
            est_bands = torch.tensor([0]) if len(self.bands) == 1 else self.bands
            est_flux = est_cat.on_fluxes[i, ecat_matches, est_bands]

            pct_err = (true_flux - est_flux) / true_flux
            # TODO: currently only supports single band
            flux_err = torch.zeros((1, self.n_bins), dtype=torch.float, device=self.device)  
            flux_err = flux_err.scatter_add(1, bins.reshape(1, -1), pct_err.reshape(1, -1))
            self.flux_err += flux_err
            self.n_matches += bins.bincount(minlength=self.n_bins)


    def compute(self):
        final_idx = -1 if self.exclude_last_bin else None
        flux_err = self.flux_err[:, :final_idx]
        n_matches = self.n_matches[:final_idx]

        total_err = flux_err.sum(dim=1) / n_matches.sum()
        binned_err = flux_err / n_matches
        results = {}
        for i, band in enumerate(self.survey_bands):
            results[f"{band}_flux_mape"] = total_err[i]
            results[f"binned_{band}_flux_mape"] = binned_err[i]

        return results


class GalaxyShapeError(Metric):
    """Compute metrics for galaxy shape parameters.

    The metrics are computed over both magnitude and parameter bins. E.g. if the magnitude bins are
    [18, 20, 22] and the parameter bins are [1, 2, 3], we first filter by magnitude range
    (i.e. <18, 18-20, 20-22, >22), and then compute metrics over each parameter bin in that
    magnitude range.
    """
    GALSIM_NAMES = [
        "disk_frac", "beta_radians", "disk_q", "a_d", "bulge_q", "a_b", "disk_hlr", "bulge_hlr"
    ]

    def __init__(
            self,
            mag_bin_cutoffs,
            param_bin_cutoffs,
            mag_band=2,
            exclude_last_bin=False,
        ):
        """Initialize state and variables.

        Args:
            mag_bin_cutoffs (list): List of magnitudes to bin by.
            param_bin_cutoffs (dict): dictionary mapping parameter names in GALSIM_NAMES to a list
                of bins for that parameter.
            mag_band (int, optional): Band to bin by. Defaults to 2.
            exclude_last_bin (bool, optional): Used to ignore the last magnitude band (e.g. if you
                want to ignore magnitude greater than some value). Defaults to False.
        """
        super().__init__()

        self.mag_band = mag_band
        self.mag_bin_cutoffs = mag_bin_cutoffs
        self.n_mag_bins = len(self.mag_bin_cutoffs) + 1
        self.exclude_last_mag_bin = exclude_last_bin  # used to ignore dim objects

        # assume all values from GALSIM_NAMES in param_bin_cutoffs
        self.param_bin_cutoffs = param_bin_cutoffs

        for param in self.GALSIM_NAMES:
            n_param_bins = len(self.param_bin_cutoffs[param]) + 1
            error_init = torch.zeros((self.n_mag_bins, n_param_bins))
            self.add_state(f"{param}_err", default=error_init.clone(), dist_reduce_fx="sum")
            self.add_state(f"n_gal_{param}", default=error_init.clone(), dist_reduce_fx="sum")

    def update(self, true_cat, est_cat, matching):
        true_mags = true_cat.magnitudes[:, :, self.mag_band].contiguous()
        mag_boundaries = torch.tensor(self.mag_bin_cutoffs, device=self.device)
        param_boundaries = {
            key: torch.tensor(val, device=self.device)
            for key, val in self.param_bin_cutoffs.items()
        }

        for i in range(true_cat.batch_size):
            tcat_matches, ecat_matches = matching[i]
            n_true = true_cat.n_sources[i].sum().item()

            true_gal = true_cat.galaxy_bools[i][tcat_matches].reshape(-1)
            true_matched_mags = true_mags[i, 0:n_true][tcat_matches]
            true_gal_mags = true_matched_mags[true_gal]

            # get magnitude bin for each matched galaxy
            mag_bins = torch.bucketize(true_gal_mags, mag_boundaries)

            true_gp = true_cat["galaxy_params"][i, tcat_matches][true_gal]
            est_gp = est_cat["galaxy_params"][i, ecat_matches][true_gal]
            gp_errors = (true_gp - est_gp) / true_gp

            for j, param in enumerate(self.GALSIM_NAMES):
                p_err = gp_errors[:, j].contiguous()
                if param == "beta_radians":
                    p_err = p_err % torch.pi

                # get parameter bin for each source
                param_bins = torch.bucketize(p_err, param_boundaries[param])

                param_errors = getattr(self, f"{param}_err")
                param_true_gal = getattr(self, f"n_gal_{param}")

                # update errors and # galaxies for [mag_bin, param_bin]
                for k in range(mag_bins.shape[0]):
                    param_errors[mag_bins[k], param_bins[k]] += p_err[k]
                    param_true_gal[mag_bins[k], param_bins[k]] += 1

                setattr(self, f"{param}_err", param_errors)
                setattr(self, f"n_gal_{param}", param_true_gal)

    def compute(self):
        results = {}
        final_idx = -1 if self.exclude_last_mag_bin else None
        for param in self.GALSIM_NAMES:
            param_err = getattr(self, f"{param}_err")[:final_idx]  # mag bins x param_bins
            n_true_gal = getattr(self, f"n_gal_{param}")[:final_idx]

            total_err = param_err.sum() / n_true_gal.sum()
            mag_binned_err = param_err.sum(dim=1) / n_true_gal.sum(dim=1)
            param_binned_err = param_err / n_true_gal

            for i in range(param_err.shape[0]):
                results[f"{param}_mape"] = total_err
                results[f"{param}_mape_mag_{i}"] = mag_binned_err[i]
                results[f"binned_{param}_mape_mag_{i}"] = param_binned_err[i]

        return results
