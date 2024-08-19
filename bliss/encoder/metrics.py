from abc import ABC, abstractmethod
from typing import List

import torch
from einops import rearrange
from scipy.optimize import linear_sum_assignment
from torchmetrics import Metric

from bliss.catalog import FullCatalog, convert_flux_to_magnitude


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
            true_mags = true_cat.on_magnitudes(c=1)[:, :, self.mag_band]
            est_mags = est_cat.on_magnitudes(c=1)[:, :, self.mag_band]

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


# it can handle the blendedness bin
class GeneralBinMetric(Metric):
    def __init__(self, bin_cutoffs: list, exclude_last_bin: bool):
        super().__init__()

        self.bin_cutoffs = bin_cutoffs
        self.exclude_last_bin = exclude_last_bin
        self.n_bins = len(self.bin_cutoffs) + 1

    def add_bin_state(self, name, additional_dim=(), dist_reduce_fx="sum"):
        default_values = torch.zeros(additional_dim + (self.n_bins,))
        self.add_state(name, default=default_values, dist_reduce_fx=dist_reduce_fx)

    def bucketize(self, value: torch.Tensor):
        cutoffs = torch.tensor(self.bin_cutoffs, device=self.device)
        return torch.bucketize(value, cutoffs)

    def bincount(self, value: torch.Tensor):
        return value.bincount(minlength=self.n_bins)

    def get_state_for_report(self, state_name):
        state = getattr(self, state_name)
        return state[..., :-1] if self.exclude_last_bin else state

    def get_report_bins(self):
        report_bins = torch.tensor(self.bin_cutoffs)
        return report_bins[:-1] if self.exclude_last_bin else report_bins


class FluxBinMetric(GeneralBinMetric):
    def __init__(self, base_njy_bin_cutoffs: list, report_bin_unit: str, exclude_last_bin: bool):
        super().__init__(bin_cutoffs=base_njy_bin_cutoffs, exclude_last_bin=exclude_last_bin)

        self.base_njy_bin_cutoffs = self.bin_cutoffs
        self.report_bin_unit = report_bin_unit
        assert self.report_bin_unit in {"njy", "ab_mag"}, "invalid bin type"

    def get_state_for_report(self, state_name):
        state = getattr(self, state_name)
        match self.report_bin_unit:
            case "njy":
                pass
            case "ab_mag":
                state = torch.flip(state, dims=(-1,))
            case _:
                raise NotImplementedError()
        return state[..., :-1] if self.exclude_last_bin else state

    def get_report_bins(self):
        report_bins = torch.tensor(self.base_njy_bin_cutoffs)
        match self.report_bin_unit:
            case "njy":
                pass
            case "ab_mag":
                report_bins = convert_flux_to_magnitude(report_bins, c=3631)
                report_bins = torch.flip(report_bins, dims=(-1,))
            case _:
                raise NotImplementedError()
        return report_bins[:-1] if self.exclude_last_bin else report_bins


class FluxBinMetricWithFilter(FluxBinMetric):
    def __init__(
        self,
        base_njy_bin_cutoffs: list,
        report_bin_unit: str,
        exclude_last_bin: bool,
        filter_list: List[CatFilter],
    ):
        super().__init__(base_njy_bin_cutoffs, report_bin_unit, exclude_last_bin)

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


class DetectionPerformance(FluxBinMetricWithFilter):
    def __init__(
        self,
        base_njy_bin_cutoffs: list = None,
        ref_band: int = 2,
        report_bin_unit: str = "njy",
        exclude_last_bin: bool = False,
        filter_list: List[CatFilter] = None,
    ):
        super().__init__(
            base_njy_bin_cutoffs if base_njy_bin_cutoffs else [],
            report_bin_unit,
            exclude_last_bin,
            filter_list if filter_list else [NullFilter()],
        )

        self.ref_band = ref_band

        detection_metrics = [
            "n_true_sources",
            "n_est_sources",
            "n_true_matches",
            "n_est_matches",
        ]
        for metric in detection_metrics:
            self.add_bin_state(metric)

    def update(self, true_cat, est_cat, matching):
        assert isinstance(true_cat, FullCatalog), "true_cat should be FullCatalog"
        assert isinstance(est_cat, FullCatalog), "est_cat should be FullCatalog"

        if self.ref_band is not None:
            true_njy_fluxes = true_cat.on_fluxes
            true_njy_fluxes = true_njy_fluxes[:, :, self.ref_band].contiguous()
            est_njy_fluxes = est_cat.on_fluxes
            est_njy_fluxes = est_njy_fluxes[:, :, self.ref_band].contiguous()
        else:
            # hack to match regardless of magnitude; intended for
            # catalogs from surveys with incompatible filter bands
            true_njy_fluxes = torch.ones_like(true_cat["plocs"][:, :, 0])
            est_njy_fluxes = torch.ones_like(est_cat["plocs"][:, :, 0])

        true_filter_bools, est_filter_bools = self.get_filter_bools(true_cat, est_cat)

        for i in range(true_cat.batch_size):
            tcat_matches, ecat_matches = matching[i]
            error_msg = "tcat_matches and ecat_matches should be of the same size"
            assert len(tcat_matches) == len(ecat_matches), error_msg
            tcat_matches, ecat_matches = tcat_matches.to(device=self.device), ecat_matches.to(
                device=self.device
            )
            n_true = true_cat["n_sources"][i].sum().item()
            n_est = est_cat["n_sources"][i].sum().item()

            cur_batch_true_njy_fluxes = true_njy_fluxes[i, :n_true]
            cur_batch_est_njy_fluxes = est_njy_fluxes[i, :n_est]

            cur_batch_true_filter_bools = true_filter_bools[i, :n_true]
            cur_batch_est_filter_bools = est_filter_bools[i, :n_est]

            tmi = cur_batch_true_njy_fluxes[cur_batch_true_filter_bools]
            emi = cur_batch_est_njy_fluxes[cur_batch_est_filter_bools]

            tcat_matches = tcat_matches[cur_batch_true_filter_bools[tcat_matches]]
            ecat_matches = ecat_matches[cur_batch_est_filter_bools[ecat_matches]]

            tmim = cur_batch_true_njy_fluxes[tcat_matches]
            emim = cur_batch_est_njy_fluxes[ecat_matches]

            self.n_true_sources += self.bincount(self.bucketize(tmi))
            self.n_est_sources += self.bincount(self.bucketize(emi))
            self.n_true_matches += self.bincount(self.bucketize(tmim))
            self.n_est_matches += self.bincount(self.bucketize(emim))

    def compute(self):
        n_est_matches = self.get_state_for_report("n_est_matches")
        n_true_matches = self.get_state_for_report("n_true_matches")
        n_est_sources = self.get_state_for_report("n_est_sources")
        n_true_sources = self.get_state_for_report("n_true_sources")

        precision_per_bin = (n_est_matches / n_est_sources).nan_to_num(0)
        recall_per_bin = (n_true_matches / n_true_sources).nan_to_num(0)
        f1_per_bin = (
            2 * precision_per_bin * recall_per_bin / (precision_per_bin + recall_per_bin)
        ).nan_to_num(0)

        precision = (n_est_matches.sum() / n_est_sources.sum()).nan_to_num(0)
        recall = (n_true_matches.sum() / n_true_sources.sum()).nan_to_num(0)
        f1 = (2 * precision * recall / (precision + recall)).nan_to_num(0)

        precision_bin_results = {
            f"detection_precision{self.postfix_str}_bin_{i}": bin_precision
            for i, bin_precision in enumerate(precision_per_bin)
        }
        recall_bin_results = {
            f"detection_recall{self.postfix_str}_bin_{i}": bin_recall
            for i, bin_recall in enumerate(recall_per_bin)
        }
        f1_bin_results = {
            f"detection_f1{self.postfix_str}_bin_{i}": bin_f1 for i, bin_f1 in enumerate(f1_per_bin)
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
        n_true_matches = self.get_state_for_report("n_true_matches")
        n_true_sources = self.get_state_for_report("n_true_sources")
        n_est_matches = self.get_state_for_report("n_est_matches")
        n_est_sources = self.get_state_for_report("n_est_sources")

        recall = (n_true_matches / n_true_sources).nan_to_num(0)
        precision = (n_est_matches / n_est_sources).nan_to_num(0)
        f1 = (2 * precision * recall / (precision + recall)).nan_to_num(0)

        return {
            f"detection_precision{self.postfix_str}": precision,
            f"detection_recall{self.postfix_str}": recall,
            f"detection_f1{self.postfix_str}": f1,
        }

    def get_internal_states(self):
        n_true_sources = self.get_state_for_report("n_true_sources")
        n_est_sources = self.get_state_for_report("n_est_sources")
        n_true_matches = self.get_state_for_report("n_true_matches")
        n_est_matches = self.get_state_for_report("n_est_matches")

        return {
            f"n_true_sources{self.postfix_str}": n_true_sources,
            f"n_est_sources{self.postfix_str}": n_est_sources,
            f"n_true_matches{self.postfix_str}": n_true_matches,
            f"n_est_matches{self.postfix_str}": n_est_matches,
        }


class SourceTypeAccuracy(FluxBinMetricWithFilter):
    def __init__(
        self,
        base_njy_bin_cutoffs: list,
        ref_band: int = 2,
        report_bin_unit: str = "njy",
        exclude_last_bin: bool = False,
        filter_list: List[CatFilter] = None,
    ):
        super().__init__(
            base_njy_bin_cutoffs,
            report_bin_unit,
            exclude_last_bin,
            filter_list if filter_list else [NullFilter()],
        )
        assert self.base_njy_bin_cutoffs, "cutoffs can't be None or empty"

        self.ref_band = ref_band

        self.add_bin_state("gal_tp")
        self.add_bin_state("gal_fp")
        self.add_bin_state("star_tp")
        self.add_bin_state("star_fp")
        self.add_bin_state("n_matches")

    def update(self, true_cat, est_cat, matching):
        true_njy_fluxes = true_cat.on_fluxes[:, :, self.ref_band].contiguous()

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

            cur_batch_true_njy_fluxes = true_njy_fluxes[i][tcat_matches]
            bin_indexes = self.bucketize(cur_batch_true_njy_fluxes)
            self.n_matches += self.bincount(bin_indexes)

            true_gal = true_cat.galaxy_bools[i][tcat_matches, 0]
            est_gal = est_cat.galaxy_bools[i][ecat_matches, 0]

            gal_tp_bool = true_gal & est_gal
            gal_fp_bool = ~true_gal & est_gal
            star_tp_bool = ~true_gal & ~est_gal
            star_fp_bool = true_gal & ~est_gal

            self.gal_tp += self.bincount(bin_indexes[gal_tp_bool])
            self.gal_fp += self.bincount(bin_indexes[gal_fp_bool])
            self.star_tp += self.bincount(bin_indexes[star_tp_bool])
            self.star_fp += self.bincount(bin_indexes[star_fp_bool])

    def compute(self):
        n_matches = self.get_state_for_report("n_matches")
        gal_tp = self.get_state_for_report("gal_tp")
        gal_fp = self.get_state_for_report("gal_fp")
        star_tp = self.get_state_for_report("star_tp")
        star_fp = self.get_state_for_report("star_fp")

        acc = ((gal_tp.sum() + star_tp.sum()) / n_matches.sum()).nan_to_num(0)
        acc_per_bin = ((gal_tp + star_tp) / n_matches).nan_to_num(0)

        star_acc = (star_tp.sum() / (n_matches.sum() - gal_tp.sum() - star_fp.sum())).nan_to_num(0)
        star_acc_per_bin = (star_tp / (n_matches - gal_tp - star_fp)).nan_to_num(0)

        gal_acc = (gal_tp.sum() / (n_matches.sum() - star_tp.sum() - gal_fp.sum())).nan_to_num(0)
        gal_acc_per_bin = (gal_tp / (n_matches - star_tp - gal_fp)).nan_to_num(0)

        acc_per_bin_results = {
            f"classification_acc{self.postfix_str}_bin_{i}": bin_acc
            for i, bin_acc in enumerate(acc_per_bin)
        }

        star_acc_per_bin_results = {
            f"classification_acc_star{self.postfix_str}_bin_{i}": bin_star_acc
            for i, bin_star_acc in enumerate(star_acc_per_bin)
        }

        gal_acc_per_bin_results = {
            f"classification_acc_galaxy{self.postfix_str}_bin_{i}": bin_gal_acc
            for i, bin_gal_acc in enumerate(gal_acc_per_bin)
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
        n_matches = self.get_state_for_report("n_matches")
        gal_tp = self.get_state_for_report("gal_tp")
        gal_fp = self.get_state_for_report("gal_fp")
        star_tp = self.get_state_for_report("star_tp")
        star_fp = self.get_state_for_report("star_fp")

        acc = ((gal_tp + star_tp) / n_matches).nan_to_num(0)
        star_acc = (star_tp / (n_matches - gal_tp - star_fp)).nan_to_num(0)
        gal_acc = (gal_tp / (n_matches - star_tp - gal_fp)).nan_to_num(0)

        return {
            f"classification_acc{self.postfix_str}": acc,
            f"classification_acc_star{self.postfix_str}": star_acc,
            f"classification_acc_galaxy{self.postfix_str}": gal_acc,
        }

    def get_internal_states(self):
        n_matches = self.get_state_for_report("n_matches")
        gal_tp = self.get_state_for_report("gal_tp")
        gal_fp = self.get_state_for_report("gal_fp")
        star_tp = self.get_state_for_report("star_tp")
        star_fp = self.get_state_for_report("star_fp")

        return {
            f"n_matches{self.postfix_str}": n_matches,
            f"gal_tp{self.postfix_str}": gal_tp,
            f"gal_fp{self.postfix_str}": gal_fp,
            f"star_tp{self.postfix_str}": star_tp,
            f"star_fp{self.postfix_str}": star_fp,
        }


class FluxError(FluxBinMetric):
    def __init__(
        self,
        survey_bands,
        base_njy_bin_cutoffs: list,
        ref_band: int = 2,
        report_bin_unit: str = "ab_mag",
        exclude_last_bin: bool = False,
    ):
        super().__init__(base_njy_bin_cutoffs, report_bin_unit, exclude_last_bin)
        self.survey_bands = survey_bands  # list of band names (e.g. "r")
        self.ref_band = ref_band

        additional_dim = (len(self.survey_bands),)
        self.add_bin_state("flux_abs_err", additional_dim=additional_dim)
        self.add_bin_state("flux_pct_err", additional_dim=additional_dim)
        self.add_bin_state("flux_abs_pct_err", additional_dim=additional_dim)
        self.add_bin_state("n_matches")

    def update(self, true_cat, est_cat, matching):
        true_njy_fluxes = true_cat.on_fluxes[:, :, self.ref_band].contiguous()

        for i in range(true_cat.batch_size):
            tcat_matches, ecat_matches = matching[i]
            n_true = true_cat["n_sources"][i].int().sum().item()
            true_matched_njy_fluxes = true_njy_fluxes[i, 0:n_true][tcat_matches].contiguous()
            bins = self.bucketize(true_matched_njy_fluxes)

            true_flux = true_cat.on_fluxes[i, tcat_matches]
            est_flux = est_cat.on_fluxes[i, ecat_matches]

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
        flux_abs_err = self.get_state_for_report("flux_abs_err")
        flux_pct_err = self.get_state_for_report("flux_pct_err")
        flux_abs_pct_err = self.get_state_for_report("flux_abs_pct_err")
        n_matches = self.get_state_for_report("n_matches")

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


class GalaxyShapeError(FluxBinMetric):
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
        base_njy_bin_cutoffs,
        ref_band=2,
        report_bin_unit="ab_mag",
        exclude_last_bin=False,
    ):
        super().__init__(base_njy_bin_cutoffs, report_bin_unit, exclude_last_bin)

        self.ref_band = ref_band

        additional_dim = (len(self.galaxy_params),)
        self.add_bin_state("galaxy_param_err", additional_dim=additional_dim)
        self.add_bin_state("disk_hlr_err")
        self.add_bin_state("bulge_hlr_err")
        self.add_bin_state("n_true_galaxies")

    def update(self, true_cat, est_cat, matching):
        true_njy_fluxes = true_cat.on_fluxes[:, :, self.ref_band].contiguous()

        for i in range(true_cat.batch_size):
            tcat_matches, ecat_matches = matching[i]
            n_true = true_cat["n_sources"][i].sum().item()

            is_gal = true_cat.galaxy_bools[i][tcat_matches][:, 0]
            # Skip if no galaxies in this image
            if (~is_gal).all():
                continue
            true_matched_njy_fluxes = true_njy_fluxes[i, 0:n_true][tcat_matches]
            true_gal_njy_fluxes = true_matched_njy_fluxes[is_gal]

            # get magnitude bin for each matched galaxy
            njy_flux_bins = self.bucketize(true_gal_njy_fluxes)
            self.n_true_galaxies += self.bincount(njy_flux_bins)

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
                self.galaxy_param_err[j] += tmp.scatter_add(0, njy_flux_bins, abs_res)

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
            self.disk_hlr_err += tmp.scatter_add(0, njy_flux_bins, abs_disk_hlr_res)

            abs_bulge_hlr_res = (true_bulge_hlr - est_bulge_hlr).abs()
            tmp = torch.zeros(self.n_bins, dtype=torch.float, device=self.device)
            self.bulge_hlr_err += tmp.scatter_add(0, njy_flux_bins, abs_bulge_hlr_res)

    def compute(self):
        galaxy_param_err = self.get_state_for_report("galaxy_param_err")
        disk_hlr_err = self.get_state_for_report("disk_hlr_err")
        bulge_hlr_err = self.get_state_for_report("bulge_hlr_err")
        n_galaxies = self.get_state_for_report("n_true_galaxies")

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
