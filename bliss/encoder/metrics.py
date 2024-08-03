from abc import ABC, abstractmethod
from typing import List

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
            true_mags = true_cat.on_mag[:, :, self.mag_band]
            est_mags = est_cat.on_mag[:, :, self.mag_band]

        matching = []
        for i in range(true_cat.batch_size):
            n_true = int(true_cat["n_sources"][i].int().sum().item())
            n_est = int(est_cat["n_sources"][i].int().sum().item())

            true_locs = true_cat["plocs"][i, :n_true]
            est_locs = est_cat["plocs"][i, :n_est]

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


class CatFilter(ABC):
    @abstractmethod
    def get_cur_filter_bools(self, true_cat, est_cat):
        """Get filter bools."""

    @abstractmethod
    def get_cur_postfix(self):
        """Get postfix for the output metric name."""


class NullFilter(CatFilter):
    def get_cur_filter_bools(self, true_cat, est_cat):
        true_filter_bools = torch.ones_like(true_cat.star_bools.squeeze(2)).bool()
        est_filter_bools = torch.ones_like(est_cat.star_bools.squeeze(2)).bool()

        return true_filter_bools, est_filter_bools

    def get_cur_postfix(self):
        return ""


class SourceTypeFilter(CatFilter):
    def __init__(self, filter_type: str) -> None:
        super().__init__()

        self.filter_type = filter_type
        assert filter_type in {"star", "galaxy"}, "invalid source_type_filter"

    def get_cur_filter_bools(self, true_cat, est_cat):
        if self.filter_type == "star":
            true_filter_bools = true_cat.star_bools.squeeze(2)
            est_filter_bools = est_cat.star_bools.squeeze(2)
        elif self.filter_type == "galaxy":
            true_filter_bools = true_cat.galaxy_bools.squeeze(2)
            est_filter_bools = est_cat.galaxy_bools.squeeze(2)
        else:
            raise NotImplementedError()

        return true_filter_bools, est_filter_bools

    def get_cur_postfix(self):
        return self.filter_type


class FilterMetric(Metric):
    def __init__(self, filter_list: List[CatFilter]):
        super().__init__()

        self.filter_list = filter_list
        assert self.filter_list, "filter_list can't be empty"
        self.postfix_str = self._get_postfix()

    def get_filter_bools(self, true_cat, est_cat):
        true_filter_bools, est_filter_bools = None, None
        for cur_filter in self.filter_list:
            if true_filter_bools is None or est_filter_bools is None:
                true_filter_bools, est_filter_bools = cur_filter.get_cur_filter_bools(
                    true_cat, est_cat
                )
            else:
                cur_true_filter_bools, cur_est_filter_bools = cur_filter.get_cur_filter_bools(
                    true_cat, est_cat
                )
                true_filter_bools &= cur_true_filter_bools
                est_filter_bools &= cur_est_filter_bools

        return true_filter_bools, est_filter_bools

    def _get_postfix(self):
        postfix_list = []
        for cur_filter in self.filter_list:
            cur_postfix = cur_filter.get_cur_postfix()
            if cur_postfix:
                postfix_list.append(cur_postfix)

        if postfix_list:
            return "_" + "_".join(postfix_list)

        return ""


class DetectionPerformance(FilterMetric):
    def __init__(
        self,
        bin_cutoffs: list = None,
        ref_band: int = 2,
        bin_type: str = "nmgy",
        exclude_last_bin: bool = False,
        filter_list: List[CatFilter] = None,
    ):
        super().__init__(filter_list if filter_list else [NullFilter()])

        self.bin_cutoffs = bin_cutoffs if bin_cutoffs else []
        self.ref_band = ref_band
        self.bin_type = bin_type
        self.exclude_last_bin = exclude_last_bin

        assert self.bin_type in {"mag", "nmgy", "njymag"}, "invalid bin type"

        detection_metrics = [
            "n_true_sources",
            "n_est_sources",
            "n_true_matches",
            "n_est_matches",
        ]
        for metric in detection_metrics:
            n_bins = len(self.bin_cutoffs) + 1  # fencepost
            init_val = torch.zeros(n_bins)
            self.add_state(metric, default=init_val, dist_reduce_fx="sum")

    def update(self, true_cat, est_cat, matching):
        assert isinstance(true_cat, FullCatalog), "true_cat should be FullCatalog"
        assert isinstance(est_cat, FullCatalog), "est_cat should be FullCatalog"

        if self.ref_band is not None:
            true_bin_measures = true_cat.on_fluxes(self.bin_type)
            true_bin_measures = true_bin_measures[:, :, self.ref_band].contiguous()
            est_bin_measures = est_cat.on_fluxes(self.bin_type)
            est_bin_measures = est_bin_measures[:, :, self.ref_band].contiguous()
        else:
            # hack to match regardless of magnitude; intended for
            # catalogs from surveys with incompatible filter bands
            true_bin_measures = torch.ones_like(true_cat["plocs"][:, :, 0])
            est_bin_measures = torch.ones_like(est_cat["plocs"][:, :, 0])

        true_filter_bools, est_filter_bools = self.get_filter_bools(true_cat, est_cat)

        cutoffs = torch.tensor(self.bin_cutoffs, device=self.device)
        n_bins = len(cutoffs) + 1

        for i in range(true_cat.batch_size):
            tcat_matches, ecat_matches = matching[i]
            error_msg = "tcat_matches and ecat_matches should be of the same size"
            assert len(tcat_matches) == len(ecat_matches), error_msg
            tcat_matches, ecat_matches = tcat_matches.to(device=self.device), ecat_matches.to(
                device=self.device
            )
            n_true = true_cat["n_sources"][i].sum().item()
            n_est = est_cat["n_sources"][i].sum().item()

            cur_batch_true_bin_meas = true_bin_measures[i, :n_true]
            cur_batch_est_bin_meas = est_bin_measures[i, :n_est]

            cur_batch_true_filter_bools = true_filter_bools[i, :n_true]
            cur_batch_est_filter_bools = est_filter_bools[i, :n_est]

            tmi = cur_batch_true_bin_meas[cur_batch_true_filter_bools]
            emi = cur_batch_est_bin_meas[cur_batch_est_filter_bools]

            tcat_matches = tcat_matches[cur_batch_true_filter_bools[tcat_matches]]
            ecat_matches = ecat_matches[cur_batch_est_filter_bools[ecat_matches]]

            tmim = cur_batch_true_bin_meas[tcat_matches]
            emim = cur_batch_est_bin_meas[ecat_matches]

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

        precision = (n_est_matches.sum() / n_est_sources.sum()).nan_to_num(0)
        recall = (n_true_matches.sum() / n_true_sources.sum()).nan_to_num(0)
        f1 = (2 * precision * recall / (precision + recall)).nan_to_num(0)

        precision_bin_results = {
            f"detection_precision{self.postfix_str}_bin_{i}": precision_per_bin[i]
            for i in range(len(precision_per_bin))
        }
        recall_bin_results = {
            f"detection_recall{self.postfix_str}_bin_{i}": recall_per_bin[i]
            for i in range(len(recall_per_bin))
        }
        f1_bin_results = {
            f"detection_f1{self.postfix_str}_bin_{i}": f1_per_bin[i] for i in range(len(f1_per_bin))
        }

        return {
            f"detection_precision{self.postfix_str}": precision,
            f"detection_recall{self.postfix_str}": recall,
            f"detection_f1{self.postfix_str}": f1,
            **precision_bin_results,
            **recall_bin_results,
            **f1_bin_results,
        }

    def get_results_on_per_bin(self):
        recall = (self.n_true_matches / self.n_true_sources).nan_to_num(0)
        precision = (self.n_est_matches / self.n_est_sources).nan_to_num(0)
        f1 = (2 * precision * recall / (precision + recall)).nan_to_num(0)

        return {
            f"detection_precision{self.postfix_str}": precision,
            f"detection_recall{self.postfix_str}": recall,
            f"detection_f1{self.postfix_str}": f1,
        }

    def get_internal_states(self):
        return {
            f"n_true_sources{self.postfix_str}": self.n_true_sources,
            f"n_est_sources{self.postfix_str}": self.n_est_sources,
            f"n_true_matches{self.postfix_str}": self.n_true_matches,
            f"n_est_matches{self.postfix_str}": self.n_est_matches,
        }

    def plot(self):
        # Compute recall, precision, and F1 score
        recall = (self.n_true_matches / self.n_true_sources).nan_to_num(0)
        precision = (self.n_est_matches / self.n_est_sources).nan_to_num(0)
        f1 = (2 * precision * recall / (precision + recall)).nan_to_num(0)

        xlabels = (
            ["< " + str(self.bin_cutoffs[0])]
            + [f"{self.bin_cutoffs[i + 1]}" for i in range(len(self.bin_cutoffs) - 1)]
            + ["> " + str(self.bin_cutoffs[-1])]
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
        fig_tag = f"({self.postfix_str[1:]})" if self.postfix_str else ""

        axes[1].plot(
            range(len(xlabels)),
            recall.tolist(),
            linestyle="solid",
            color=c1,
            label=f"BLISS Recall {fig_tag}",
        )
        axes[1].plot(
            range(len(xlabels)),
            precision.tolist(),
            linestyle="solid",
            color=c2,
            label=f"BLISS Precision {fig_tag}",
        )
        axes[1].plot(
            range(len(xlabels)),
            f1.tolist(),
            linestyle="solid",
            color=c3,
            label=f"BLISS F1 {fig_tag}",
        )
        axes[1].set_xlabel(self.bin_type)
        axes[1].set_xticks(range(len(xlabels)))
        axes[1].set_xticklabels(xlabels, rotation=45)
        axes[1].set_ylim([0, 1])
        axes[1].legend()

        axes[0].step(
            range(len(xlabels)),
            n_true_sources.tolist(),
            label=f"Number of true sources {fig_tag}",
            where="mid",
            color=c4,
        )
        axes[0].step(
            range(len(xlabels)),
            n_true_matches.tolist(),
            label=f"Number of BLISS matches {fig_tag}",
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

        return fig, axes


class SourceTypeAccuracy(FilterMetric):
    def __init__(
        self,
        bin_cutoffs: list,
        ref_band: int = 2,
        bin_type: str = "nmgy",
        filter_list: List[CatFilter] = None,
    ):
        super().__init__(filter_list if filter_list else [NullFilter()])

        self.bin_cutoffs = bin_cutoffs
        self.ref_band = ref_band
        self.bin_type = bin_type

        assert self.bin_cutoffs, "flux_bin_cutoffs can't be None or empty"
        assert self.bin_type in {"mag", "nmgy", "njymag"}, "invalid bin type"

        n_bins = len(self.bin_cutoffs) + 1

        self.add_state("gal_tp", default=torch.zeros(n_bins), dist_reduce_fx="sum")
        self.add_state("gal_fp", default=torch.zeros(n_bins), dist_reduce_fx="sum")
        self.add_state("star_tp", default=torch.zeros(n_bins), dist_reduce_fx="sum")
        self.add_state("star_fp", default=torch.zeros(n_bins), dist_reduce_fx="sum")
        self.add_state("n_matches", default=torch.zeros(n_bins), dist_reduce_fx="sum")

    def update(self, true_cat, est_cat, matching):
        cutoffs = torch.tensor(self.bin_cutoffs, device=self.device)
        n_bins = len(cutoffs) + 1

        true_bin_measures = true_cat.on_fluxes(self.bin_type)[:, :, self.ref_band].contiguous()

        true_filter_bools, _ = self.get_filter_bools(true_cat, est_cat)

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

            cur_batch_true_bin_meas = true_bin_measures[i][tcat_matches]
            bin_indexes = torch.bucketize(cur_batch_true_bin_meas, cutoffs)
            _, to_bin_mapping = torch.sort(bin_indexes)
            per_bin_elements_count = bin_indexes.bincount(minlength=n_bins)

            true_gal = true_cat.galaxy_bools[i][tcat_matches][to_bin_mapping]
            est_gal = est_cat.galaxy_bools[i][ecat_matches][to_bin_mapping]

            gal_tp_bool = torch.split(true_gal & est_gal, per_bin_elements_count.tolist())
            gal_fp_bool = torch.split(~true_gal & est_gal, per_bin_elements_count.tolist())
            star_tp_bool = torch.split(~true_gal & ~est_gal, per_bin_elements_count.tolist())
            star_fp_bool = torch.split(true_gal & ~est_gal, per_bin_elements_count.tolist())

            gal_tp = torch.tensor([i.sum() for i in gal_tp_bool], device=self.device)
            gal_fp = torch.tensor([i.sum() for i in gal_fp_bool], device=self.device)
            star_tp = torch.tensor([i.sum() for i in star_tp_bool], device=self.device)
            star_fp = torch.tensor([i.sum() for i in star_fp_bool], device=self.device)

            self.n_matches += per_bin_elements_count
            self.gal_tp += gal_tp
            self.gal_fp += gal_fp
            self.star_tp += star_tp
            self.star_fp += star_fp

    def compute(self):
        acc = ((self.gal_tp.sum() + self.star_tp.sum()) / self.n_matches.sum()).nan_to_num(0)
        acc_per_bin = ((self.gal_tp + self.star_tp) / self.n_matches).nan_to_num(0)

        star_acc = (
            self.star_tp.sum() / (self.n_matches.sum() - self.gal_tp.sum() - self.star_fp.sum())
        ).nan_to_num(0)
        star_acc_per_bin = (
            self.star_tp / (self.n_matches - self.gal_tp - self.star_fp)
        ).nan_to_num(0)

        gal_acc = (
            self.gal_tp.sum() / (self.n_matches.sum() - self.star_tp.sum() - self.gal_fp.sum())
        ).nan_to_num(0)
        gal_acc_per_bin = (self.gal_tp / (self.n_matches - self.star_tp - self.gal_fp)).nan_to_num(
            0
        )

        acc_per_bin_results = {
            f"classification_acc{self.postfix_str}_bin_{i}": acc_per_bin[i]
            for i in range(len(acc_per_bin))
        }

        star_acc_per_bin_results = {
            f"classification_acc_star{self.postfix_str}_bin_{i}": star_acc_per_bin[i]
            for i in range(len(star_acc_per_bin))
        }

        gal_acc_per_bin_results = {
            f"classification_acc_galaxy{self.postfix_str}_bin_{i}": gal_acc_per_bin[i]
            for i in range(len(gal_acc_per_bin))
        }

        return {
            f"classification_acc{self.postfix_str}": acc.item(),
            f"classification_acc_star{self.postfix_str}": star_acc.item(),
            f"classification_acc_galaxy{self.postfix_str}": gal_acc.item(),
            **acc_per_bin_results,
            **star_acc_per_bin_results,
            **gal_acc_per_bin_results,
        }

    def get_results_on_per_bin(self):
        acc = ((self.gal_tp + self.star_tp) / self.n_matches).nan_to_num(0)
        star_acc = (self.star_tp / (self.n_matches - self.gal_tp - self.star_fp)).nan_to_num(0)
        gal_acc = (self.gal_tp / (self.n_matches - self.star_tp - self.gal_fp)).nan_to_num(0)

        return {
            f"classification_acc{self.postfix_str}": acc,
            f"classification_acc_star{self.postfix_str}": star_acc,
            f"classification_acc_galaxy{self.postfix_str}": gal_acc,
        }

    def get_internal_states(self):
        return {
            f"n_matches{self.postfix_str}": self.n_matches,
            f"gal_tp{self.postfix_str}": self.gal_tp,
            f"gal_fp{self.postfix_str}": self.gal_fp,
            f"star_tp{self.postfix_str}": self.star_tp,
            f"star_fp{self.postfix_str}": self.star_fp,
        }


class FluxError(Metric):
    def __init__(
        self,
        survey_bands,
        bin_cutoffs: list,
        ref_band: int = 2,
        bin_type: str = "mag",
        exclude_last_bin: bool = False,
    ):
        super().__init__()
        self.survey_bands = survey_bands  # list of band names (e.g. "r")
        self.ref_band = ref_band
        self.bin_cutoffs = bin_cutoffs
        self.bin_type = bin_type
        self.exclude_last_bin = exclude_last_bin
        self.n_bins = len(self.bin_cutoffs) + 1

        self.add_state(
            "flux_abs_err",
            default=torch.zeros((len(self.survey_bands), self.n_bins)),  # n_bins per band
            dist_reduce_fx="sum",
        )
        self.add_state(
            "flux_pct_err",
            default=torch.zeros((len(self.survey_bands), self.n_bins)),  # n_bins per band
            dist_reduce_fx="sum",
        )
        self.add_state(
            "flux_abs_pct_err",
            default=torch.zeros((len(self.survey_bands), self.n_bins)),  # n_bins per band
            dist_reduce_fx="sum",
        )
        self.add_state("n_matches", default=torch.zeros(self.n_bins), dist_reduce_fx="sum")

    def update(self, true_cat, est_cat, matching):
        cutoffs = torch.tensor(self.bin_cutoffs, device=self.device)
        true_bin_measures = true_cat.on_fluxes(self.bin_type)[:, :, self.ref_band].contiguous()

        for i in range(true_cat.batch_size):
            tcat_matches, ecat_matches = matching[i]
            n_true = true_cat["n_sources"][i].int().sum().item()
            bin_measure = true_bin_measures[i, 0:n_true][tcat_matches].contiguous()
            bins = torch.bucketize(bin_measure, cutoffs)

            true_flux = true_cat.on_fluxes("nmgy")[i, tcat_matches]
            est_flux = est_cat.on_fluxes("nmgy")[i, ecat_matches]

            # Compute and update percent error per band
            abs_err = (true_flux - est_flux).abs()
            pct_err = (true_flux - est_flux) / true_flux
            abs_pct_err = pct_err.abs()
            for band in range(len(self.survey_bands)):  # noqa: WPS518
                tmp = torch.zeros((self.n_bins,), dtype=torch.float, device=self.device)
                tmp = tmp.scatter_add(0, bins.reshape(-1), abs_err[..., band].reshape(-1))
                self.flux_abs_err[band] += tmp

                tmp = torch.zeros((self.n_bins,), dtype=torch.float, device=self.device)
                tmp = tmp.scatter_add(0, bins.reshape(-1), pct_err[..., band].reshape(-1))
                self.flux_pct_err[band] += tmp

                tmp = torch.zeros((self.n_bins,), dtype=torch.float, device=self.device)
                tmp = tmp.scatter_add(0, bins.reshape(-1), abs_pct_err[..., band].reshape(-1))
                self.flux_abs_pct_err[band] += tmp.abs()
            self.n_matches += bins.bincount(minlength=self.n_bins)

    def compute(self):
        final_idx = -1 if self.exclude_last_bin else None
        flux_abs_err = self.flux_abs_err[:, :final_idx]
        flux_pct_err = self.flux_pct_err[:, :final_idx]
        flux_abs_pct_err = self.flux_abs_pct_err[:, :final_idx]
        n_matches = self.n_matches[:final_idx]

        # Compute final metrics
        mae = flux_abs_err.sum(dim=1) / n_matches.sum()
        binned_mae = flux_abs_err / n_matches
        mpe = flux_pct_err.sum(dim=1) / n_matches.sum()
        binned_mpe = flux_pct_err / n_matches
        mape = flux_abs_pct_err.sum(dim=1) / n_matches.sum()
        binned_mape = flux_abs_pct_err / n_matches

        results = {}
        for i, band in enumerate(self.survey_bands):
            results[f"flux_err_{band}_mae"] = mae[i]
            results[f"flux_err_{band}_mpe"] = mpe[i]
            results[f"flux_err_{band}_mape"] = mape[i]
            for j in range(binned_mpe.shape[1]):
                results[f"flux_err_{band}_mae_bin_{j}"] = binned_mae[i, j]
                results[f"flux_err_{band}_mpe_bin_{j}"] = binned_mpe[i, j]
                results[f"flux_err_{band}_mape_bin_{j}"] = binned_mape[i, j]

        return results


class GalaxyShapeError(Metric):
    galaxy_params = [
        "galaxy_disk_frac",
        "galaxy_beta_radians",
        "galaxy_disk_q",
        "galaxy_a_d",
        "galaxy_bulge_q",
        "galaxy_a_b",
    ]
    galaxy_param_to_idx = {param: i for i, param in enumerate(galaxy_params)}

    def __init__(
        self,
        bin_cutoffs,
        ref_band=2,
        bin_type="mag",
        exclude_last_bin=False,
    ):
        super().__init__()

        self.ref_band = ref_band
        self.bin_cutoffs = bin_cutoffs
        self.n_bins = len(self.bin_cutoffs) + 1
        self.bin_type = bin_type
        self.exclude_last_bin = exclude_last_bin  # used to ignore dim objects

        gpe_init = torch.zeros((len(self.galaxy_params), self.n_bins))
        self.add_state("galaxy_param_err", default=gpe_init, dist_reduce_fx="sum")
        self.add_state("disk_hlr_err", torch.zeros(self.n_bins), dist_reduce_fx="sum")
        self.add_state("bulge_hlr_err", torch.zeros(self.n_bins), dist_reduce_fx="sum")
        self.add_state("n_true_galaxies", default=torch.zeros(self.n_bins), dist_reduce_fx="sum")

    def update(self, true_cat, est_cat, matching):
        true_bin_meas = true_cat.on_fluxes(self.bin_type)[:, :, self.ref_band].contiguous()
        cutoffs = torch.tensor(self.bin_cutoffs, device=self.device)

        for i in range(true_cat.batch_size):
            tcat_matches, ecat_matches = matching[i]
            n_true = true_cat["n_sources"][i].sum().item()

            is_gal = true_cat.galaxy_bools[i][tcat_matches][:, 0]
            # Skip if no galaxies in this image
            if (~is_gal).all():
                continue
            true_matched_mags = true_bin_meas[i, 0:n_true][tcat_matches]
            true_gal_mags = true_matched_mags[is_gal]

            # get magnitude bin for each matched galaxy
            mag_bins = torch.bucketize(true_gal_mags, cutoffs)
            self.n_true_galaxies += mag_bins.bincount(minlength=self.n_bins)

            true_gal_params = true_cat["galaxy_params"][i, tcat_matches][is_gal]

            for j, name in enumerate(self.galaxy_params):
                true_param = true_gal_params[:, j]
                est_param = est_cat[name][i, ecat_matches][is_gal, 0]
                abs_res = (true_param - est_param).abs()

                # Wrap angle around pi
                if name == "galaxy_beta_radians":
                    abs_res = abs_res % torch.pi

                # Update bins
                tmp = torch.zeros(self.n_bins, dtype=torch.float, device=self.device)
                self.galaxy_param_err[j] += tmp.scatter_add(0, mag_bins, abs_res)

            # Compute HLRs for disk and bulge
            true_a_d = true_gal_params[:, self.galaxy_param_to_idx["galaxy_a_d"]]
            true_disk_q = true_gal_params[:, self.galaxy_param_to_idx["galaxy_disk_q"]]
            true_b_d = true_a_d * true_disk_q
            true_disk_hlr = torch.sqrt(true_a_d * true_b_d)

            est_a_d = est_cat["galaxy_a_d"][i, ecat_matches][is_gal, 0]
            est_disk_q = est_cat["galaxy_disk_q"][i, ecat_matches][is_gal, 0]
            est_b_d = est_a_d * est_disk_q
            est_disk_hlr = torch.sqrt(est_a_d * est_b_d)

            true_a_b = true_gal_params[:, self.galaxy_param_to_idx["galaxy_a_b"]]
            true_bulge_q = true_gal_params[:, self.galaxy_param_to_idx["galaxy_bulge_q"]]
            true_b_b = true_a_b * true_bulge_q
            true_bulge_hlr = torch.sqrt(true_a_b * true_b_b)

            est_a_b = est_cat["galaxy_a_b"][i, ecat_matches][is_gal, 0]
            est_bulge_q = est_cat["galaxy_bulge_q"][i, ecat_matches][is_gal, 0]
            est_b_b = est_a_b * est_bulge_q
            est_bulge_hlr = torch.sqrt(est_a_b * est_b_b)

            abs_disk_hlr_res = (true_disk_hlr - est_disk_hlr).abs()
            tmp = torch.zeros(self.n_bins, dtype=torch.float, device=self.device)
            self.disk_hlr_err += tmp.scatter_add(0, mag_bins, abs_disk_hlr_res)

            abs_bulge_hlr_res = (true_bulge_hlr - est_bulge_hlr).abs()
            tmp = torch.zeros(self.n_bins, dtype=torch.float, device=self.device)
            self.bulge_hlr_err += tmp.scatter_add(0, mag_bins, abs_bulge_hlr_res)

    def compute(self):
        final_idx = -1 if self.exclude_last_bin else None
        galaxy_param_err = self.galaxy_param_err[:, :final_idx]
        disk_hlr_err = self.disk_hlr_err[:final_idx]
        bulge_hlr_err = self.bulge_hlr_err[:final_idx]
        n_galaxies = self.n_true_galaxies[:final_idx]

        gal_param_mae = galaxy_param_err.sum(dim=1) / n_galaxies.sum()
        binned_gal_param_mae = galaxy_param_err / n_galaxies

        disk_hlr_mae = disk_hlr_err.sum() / n_galaxies.sum()
        binned_disk_hlr_mae = disk_hlr_err / n_galaxies
        bulge_hlr_mae = bulge_hlr_err.sum() / n_galaxies.sum()
        binned_bulge_hlr_mae = bulge_hlr_err / n_galaxies

        results = {}
        for i, name in enumerate(self.galaxy_params):
            results[f"{name}_mae"] = gal_param_mae[i]
            for j in range(self.n_bins):
                results[f"{name}_mae_bin_{j}"] = binned_gal_param_mae[i, j]

        results["galaxy_disk_hlr_mae"] = disk_hlr_mae
        results["galaxy_bulge_hlr_mae"] = bulge_hlr_mae
        for j in range(self.n_bins):
            results[f"galaxy_disk_hlr_mae_bin_{j}"] = binned_disk_hlr_mae[j]
            results[f"galaxy_bulge_hlr_mae_bin_{j}"] = binned_bulge_hlr_mae[j]

        return results
