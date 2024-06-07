import numpy as np
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


class SourceFilterMetric(Metric):
    def __init__(self, source_type_filter: str):
        super().__init__()

        self.source_type_filter = source_type_filter
        assert source_type_filter in {"total", "star", "galaxy"}, "invalid source_type_filter"

    def _get_filter_bools(self, true_cat, est_cat):
        if self.source_type_filter == "star":
            true_filter_bools = true_cat.star_bools.squeeze(2)
            est_filter_bools = est_cat.star_bools.squeeze(2)
        elif self.source_type_filter == "galaxy":
            true_filter_bools = true_cat.galaxy_bools.squeeze(2)
            est_filter_bools = est_cat.galaxy_bools.squeeze(2)
        elif self.source_type_filter == "total":
            true_filter_bools = torch.ones_like(true_cat.star_bools.squeeze(2)).bool()
            est_filter_bools = torch.ones_like(est_cat.star_bools.squeeze(2)).bool()
        else:
            raise NotImplementedError()

        return true_filter_bools, est_filter_bools


class DetectionPerformance(SourceFilterMetric):
    """Calculates detection and classification metrics between two catalogs.
    Note that galaxy classification metrics are only computed when the values are available in
    both catalogs.
    """

    def __init__(
        self,
        mag_band: int = 2,
        mag_bin_cutoffs: list = None,
        exclude_last_bin: bool = False,
        source_type_filter: str = "total",
        bin_unit_is_flux: bool = False,
    ):
        super().__init__(source_type_filter)

        self.mag_band = mag_band
        self.mag_bin_cutoffs = mag_bin_cutoffs if mag_bin_cutoffs else []
        self.exclude_last_bin = exclude_last_bin
        self.bin_unit_is_flux = bin_unit_is_flux

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
        assert isinstance(true_cat, FullCatalog), "true_cat should be FullCatalog"
        assert isinstance(est_cat, FullCatalog), "est_cat should be FullCatalog"

        if self.mag_band is not None:
            unit = "on_fluxes" if self.bin_unit_is_flux else "magnitudes"
            true_mags = true_cat[unit][:, :, self.mag_band].contiguous()
            est_mags = est_cat[unit][:, :, self.mag_band].contiguous()
        else:
            # hack to match regardless of magnitude; intended for
            # catalogs from surveys with incompatible filter bands
            true_mags = torch.ones_like(true_cat.plocs[:, :, 0])
            est_mags = torch.ones_like(est_cat.plocs[:, :, 0])

        true_filter_bools, est_filter_bools = self._get_filter_bools(true_cat, est_cat)

        cutoffs = torch.tensor(self.mag_bin_cutoffs, device=self.device)
        n_bins = len(cutoffs) + 1

        for i in range(true_cat.batch_size):
            tcat_matches, ecat_matches = matching[i]
            error_msg = "tcat_matches and ecat_matches should be of the same size"
            assert len(tcat_matches) == len(ecat_matches), error_msg
            tcat_matches, ecat_matches = tcat_matches.to(device=self.device), ecat_matches.to(
                device=self.device
            )
            n_true = true_cat.n_sources[i].sum().item()
            n_est = est_cat.n_sources[i].sum().item()

            cur_batch_true_mags = true_mags[i, :n_true]
            cur_batch_est_mags = est_mags[i, :n_est]

            cur_batch_true_filter_bools = true_filter_bools[i, :n_true]
            cur_batch_est_filter_bools = est_filter_bools[i, :n_est]

            tmi = cur_batch_true_mags[cur_batch_true_filter_bools]
            emi = cur_batch_est_mags[cur_batch_est_filter_bools]

            tcat_matches = tcat_matches[cur_batch_true_filter_bools[tcat_matches]]
            ecat_matches = ecat_matches[cur_batch_est_filter_bools[ecat_matches]]

            tmim = cur_batch_true_mags[tcat_matches]
            emim = cur_batch_est_mags[ecat_matches]

            self.n_true_sources += torch.bucketize(tmi, cutoffs).bincount(minlength=n_bins)
            self.n_est_sources += torch.bucketize(emi, cutoffs).bincount(minlength=n_bins)
            self.n_true_matches += torch.bucketize(tmim, cutoffs).bincount(minlength=n_bins)
            self.n_est_matches += torch.bucketize(emim, cutoffs).bincount(minlength=n_bins)

    def compute(self):
        n_est_matches = self.n_est_matches[:-1] if self.exclude_last_bin else self.n_est_matches
        n_true_matches = self.n_true_matches[:-1] if self.exclude_last_bin else self.n_true_matches
        n_est_sources = self.n_est_sources[:-1] if self.exclude_last_bin else self.n_est_sources
        n_true_sources = self.n_true_sources[:-1] if self.exclude_last_bin else self.n_true_sources

        precision_per_bin = (n_est_matches / n_est_sources).nan_to_num(0)
        recall_per_bin = (n_true_matches / n_true_sources).nan_to_num(0)
        f1_per_bin = (
            2 * precision_per_bin * recall_per_bin / (precision_per_bin + recall_per_bin)
        ).nan_to_num(0)

        precision = n_est_matches.sum() / n_est_sources.sum()
        recall = n_true_matches.sum() / n_true_sources.sum()
        f1 = 2 * precision * recall / (precision + recall)

        detection_precision_name = (
            f"detection_precision_{self.source_type_filter}"
            if self.source_type_filter != "total"
            else "detection_precision"
        )
        detection_recall_name = (
            f"detection_recall_{self.source_type_filter}"
            if self.source_type_filter != "total"
            else "detection_recall"
        )
        detection_f1_name = (
            f"detection_f1_{self.source_type_filter}"
            if self.source_type_filter != "total"
            else "detection_f1"
        )

        precision_bin_results = {
            f"{detection_precision_name}_bin_{i}": precision_per_bin[i]
            for i in range(len(precision_per_bin))
        }
        recall_bin_results = {
            f"{detection_recall_name}_bin_{i}": recall_per_bin[i]
            for i in range(len(recall_per_bin))
        }
        f1_bin_results = {
            f"{detection_f1_name}_bin_{i}": f1_per_bin[i] for i in range(len(f1_per_bin))
        }

        return {
            f"{detection_precision_name}": precision,
            f"{detection_recall_name}": recall,
            f"{detection_f1_name}": f1,
            **precision_bin_results,
            **recall_bin_results,
            **f1_bin_results,
        }

    def get_results_on_per_flux_bin(self):
        recall = (self.n_true_matches / self.n_true_sources).nan_to_num(0)
        precision = (self.n_est_matches / self.n_est_sources).nan_to_num(0)
        f1 = (2 * precision * recall / (precision + recall)).nan_to_num(0)

        if self.source_type_filter == "total":
            return {
                "detection_precision": precision,
                "detection_recall": recall,
                "detection_f1": f1,
            }

        return {
            f"detection_precision_{self.source_type_filter}": precision,
            f"detection_recall_{self.source_type_filter}": recall,
            f"detection_f1_{self.source_type_filter}": f1,
        }

    def plot(self):
        # Compute recall, precision, and F1 score
        recall = (self.n_true_matches / self.n_true_sources).nan_to_num(0)
        precision = (self.n_est_matches / self.n_est_sources).nan_to_num(0)
        f1 = (2 * precision * recall / (precision + recall)).nan_to_num(0)

        xlabels = (
            ["< " + str(self.mag_bin_cutoffs[0])]
            + [f"{self.mag_bin_cutoffs[i + 1]}" for i in range(len(self.mag_bin_cutoffs) - 1)]
            + ["> " + str(self.mag_bin_cutoffs[-1])]
        )

        n_true_sources = self.n_true_sources
        n_true_matches = self.n_true_matches

        if self.exclude_last_bin:
            precision = precision[:-1]
            recall = recall[:-1]
            f1 = f1[:-1]
            xlabels = xlabels[:-1]
            n_true_sources = n_true_sources[:-1]
            n_true_matches = n_true_matches[:-1]

        sns.set_theme(style="whitegrid")
        fig, axes = plt.subplots(
            2, 1, figsize=(10, 10), gridspec_kw={"height_ratios": [1, 2]}, sharex="col"
        )

        c1, c2, c3, c4 = plt.rcParams["axes.prop_cycle"].by_key()["color"][0:4]
        axes[1].plot(
            range(len(xlabels)),
            recall.tolist(),
            linestyle="solid",
            color=c1,
            label=f"BLISS Recall ({self.source_type_filter})",
        )
        axes[1].plot(
            range(len(xlabels)),
            precision.tolist(),
            linestyle="solid",
            color=c2,
            label=f"BLISS Precision ({self.source_type_filter})",
        )
        axes[1].plot(
            range(len(xlabels)),
            f1.tolist(),
            linestyle="solid",
            color=c3,
            label=f"BLISS F1 ({self.source_type_filter})",
        )
        axes[1].set_xlabel("Flux" if self.bin_unit_is_flux else "Magnitudes")
        axes[1].set_xticks(range(len(xlabels)))
        axes[1].set_xticklabels(xlabels, rotation=45)
        axes[1].set_ylim([0, 1])
        axes[1].legend()

        axes[0].step(
            range(len(xlabels)),
            n_true_sources.tolist(),
            label=f"# true sources ({self.source_type_filter})",
            where="mid",
            color=c4,
        )
        axes[0].step(
            range(len(xlabels)),
            n_true_matches.tolist(),
            label=f"# BLISS matches ({self.source_type_filter})",
            ls="--",
            where="mid",
            color=c4,
        )
        count_max = n_true_sources.max().item()
        count_ticks = np.round(np.linspace(0, count_max, 5), -3)
        axes[0].set_yticks(count_ticks)
        axes[0].set_ylabel("Count")
        axes[0].legend()

        plt.tight_layout()
        plt.show()

        return fig, axes


class SourceTypeAccuracy(SourceFilterMetric):
    def __init__(
        self, flux_bin_cutoffs: list, ref_band: int = 2, source_type_filter: str = "total"
    ):
        super().__init__(source_type_filter)

        self.flux_bin_cutoffs = flux_bin_cutoffs
        self.ref_band = ref_band

        assert self.flux_bin_cutoffs, "flux_bin_cutoffs can't be None or empty"

        n_bins = len(self.flux_bin_cutoffs) + 1

        self.add_state("gal_tp", default=torch.zeros(n_bins), dist_reduce_fx="sum")
        self.add_state("star_tp", default=torch.zeros(n_bins), dist_reduce_fx="sum")
        self.add_state("n_matches", default=torch.zeros(n_bins), dist_reduce_fx="sum")

    def update(self, true_cat, est_cat, matching):
        cutoffs = torch.tensor(self.flux_bin_cutoffs, device=self.device)
        n_bins = len(cutoffs) + 1

        true_fluxes = true_cat["on_fluxes"][:, :, self.ref_band].contiguous()

        true_filter_bools, _ = self._get_filter_bools(true_cat, est_cat)

        for i in range(true_cat.batch_size):
            tcat_matches, ecat_matches = matching[i]
            assert len(tcat_matches) == len(
                ecat_matches
            ), "tcat_matches and ecat_matches should be of the same size"
            if tcat_matches.shape[0] == 0 or ecat_matches.shape[0] == 0:
                continue
            tcat_matches, ecat_matches = tcat_matches.to(device=self.device), ecat_matches.to(
                device=self.device
            )
            tcat_matches_filter = true_filter_bools[i][tcat_matches]
            tcat_matches = tcat_matches[tcat_matches_filter]
            ecat_matches = ecat_matches[tcat_matches_filter]

            cur_batch_true_fluxes = true_fluxes[i][tcat_matches]
            bin_indexes = torch.bucketize(cur_batch_true_fluxes, cutoffs)
            _, to_bin_mapping = torch.sort(bin_indexes)
            per_bin_elements_count = bin_indexes.bincount(minlength=n_bins)

            true_gal = true_cat.galaxy_bools[i][tcat_matches][to_bin_mapping]
            est_gal = est_cat.galaxy_bools[i][ecat_matches][to_bin_mapping]

            gal_tp_bool = torch.split(true_gal * est_gal, per_bin_elements_count.tolist())
            star_tp_bool = torch.split(~true_gal * ~est_gal, per_bin_elements_count.tolist())

            gal_tp = torch.tensor([i.sum() for i in gal_tp_bool], device=self.device)
            star_tp = torch.tensor([i.sum() for i in star_tp_bool], device=self.device)

            self.n_matches += per_bin_elements_count
            self.gal_tp += gal_tp
            self.star_tp += star_tp

    def compute(self):
        acc = ((self.gal_tp.sum() + self.star_tp.sum()) / self.n_matches.sum()).nan_to_num(0)
        acc_per_bin = ((self.gal_tp + self.star_tp) / self.n_matches).nan_to_num(0)

        acc_name = (
            f"classification_acc_{self.source_type_filter}"
            if self.source_type_filter != "total"
            else "classification_acc"
        )

        acc_per_bin_results = {
            f"{acc_name}_bin_{i}": acc_per_bin[i] for i in range(len(acc_per_bin))
        }

        return {
            f"{acc_name}": acc.item(),
            **acc_per_bin_results,
        }

    def get_results_on_per_flux_bin(self):
        acc = ((self.gal_tp + self.star_tp) / self.n_matches).nan_to_num(0)

        if self.source_type_filter == "total":
            return {
                "classification_acc": acc,
            }

        return {
            f"classification_acc_{self.source_type_filter}": acc,
        }


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
