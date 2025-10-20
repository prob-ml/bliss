import torch
import torch.distributed
import matplotlib.pyplot as plt
import seaborn as sns

from typing import List
from collections import OrderedDict
from torchmetrics import Metric
from einops import rearrange
from scipy import stats

from bliss.catalog import FullCatalog
from bliss.encoder.metrics import CatFilter, NullFilter, convert_flux_to_magnitude


class FilterMetric(Metric):
    def __init__(
        self,
        filter_list: List[CatFilter],
    ):
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
    

class NSourcesAccuracy(Metric):
    def __init__(
        self,
        max_n_sources,
        tile_slen,
    ):
        super().__init__()

        self.max_n_sources = max_n_sources
        self.tile_slen = tile_slen

        n_sources_metrics = [
            "n_true_s",
            "n_est_s",
            "n_s_match",
        ]
        for metrics in n_sources_metrics:
            self.add_state(metrics, 
                           default=torch.zeros(self.max_n_sources + 1, dtype=torch.long), 
                           dist_reduce_fx="sum")

    def update(self, true_cat, est_cat, matching):
        assert isinstance(true_cat, FullCatalog), "true_cat should be FullCatalog"
        assert isinstance(est_cat, FullCatalog), "est_cat should be FullCatalog"
        assert hasattr(true_cat, "ori_tile_cat")
        assert hasattr(est_cat, "ori_tile_cat")

        true_tile_cat = true_cat.ori_tile_cat
        est_tile_cat = est_cat.ori_tile_cat

        true_n_sources = true_tile_cat["n_sources"]  # (b, h, w)
        est_n_sources = rearrange(est_tile_cat["n_sources_multi"],
                                  "b h w 1 1 -> b h w")

        for i in range(self.max_n_sources + 1):
            true_i = true_n_sources == i
            est_i = est_n_sources == i
            self.n_true_s[i] += true_i.sum()
            self.n_est_s[i] += est_i.sum()
            self.n_s_match[i] += (true_i & est_i).sum()

    def compute(self):
        precision = (self.n_s_match / self.n_est_s).nan_to_num(0)
        recall = (self.n_s_match / self.n_true_s).nan_to_num(0)
        f1 = (2 * precision * recall / (precision + recall)).nan_to_num(0)

        out_dict = {}
        for i in range(self.max_n_sources + 1):
            out_dict[f"n_sources:s{i}_precision"] = precision[i]
            out_dict[f"n_sources:s{i}_recall"] = recall[i]
            out_dict[f"n_sources:s{i}_f1"] = f1[i]
            out_dict[f"n_sources:n_est_s{i}"] = self.n_est_s[i].float()
            out_dict[f"n_sources:n_true_s{i}"] = self.n_true_s[i].float()
        return out_dict


class DetectionPerformance(FilterMetric):
    def __init__(
        self,
        filter_list: List[CatFilter] = None,
    ):
        super().__init__(
            filter_list if filter_list else [NullFilter()],
        )

        detection_metrics = [
            "n_true_sources",
            "n_est_sources",
            "n_true_matches",
            "n_est_matches",
        ]
        for metric in detection_metrics:
            self.add_state(metric, default=torch.zeros(1), dist_reduce_fx="sum")

    def update(self, true_cat, est_cat, matching):
        assert isinstance(true_cat, FullCatalog), "true_cat should be FullCatalog"
        assert isinstance(est_cat, FullCatalog), "est_cat should be FullCatalog"

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

            cur_batch_true_filter_bools = true_filter_bools[i, :n_true]
            cur_batch_est_filter_bools = est_filter_bools[i, :n_est]

            tmi = cur_batch_true_filter_bools.sum()
            emi = cur_batch_est_filter_bools.sum()

            tcat_matches = tcat_matches[cur_batch_true_filter_bools[tcat_matches]]
            ecat_matches = ecat_matches[cur_batch_est_filter_bools[ecat_matches]]

            tmim = tcat_matches.numel()
            emim = ecat_matches.numel()

            self.n_true_sources += tmi
            self.n_est_sources += emi
            self.n_true_matches += tmim
            self.n_est_matches += emim

    def compute(self):
        precision = (self.n_est_matches / self.n_est_sources).nan_to_num(0)
        recall = (self.n_true_matches / self.n_true_sources).nan_to_num(0)
        f1 = (2 * precision * recall / (precision + recall)).nan_to_num(0)

        return {
            f"detection_precision{self.postfix_str}": precision,
            f"detection_recall{self.postfix_str}": recall,
            f"detection_f1{self.postfix_str}": f1,
            f"n_true_sources{self.postfix_str}": self.n_true_sources,
            f"n_est_sources{self.postfix_str}": self.n_est_sources,
        }
    

def _is_global_zero():
    return (not torch.distributed.is_available()) or \
           (not torch.distributed.is_initialized()) or \
           (torch.distributed.get_rank() == 0)
    

class AsymmetricCM(Metric):
    def __init__(
        self,
        max_n_sources,
        locs_bin_boundaries,
        flux_bin_boundaries,
        flux_bands,
    ):
        super().__init__()

        self.max_n_sources = max_n_sources
        self.locs_bin_boundaries = torch.tensor(locs_bin_boundaries)
        self.flux_bin_boundaries = torch.tensor(flux_bin_boundaries)

        self.add_state("n_sources_cm",
                       default=torch.zeros(max_n_sources + 1, 
                                           max_n_sources + 1, 
                                           dtype=torch.long),
                       dist_reduce_fx="sum")
        self.add_state("locs_x_cm",
                       default=torch.zeros(len(self.locs_bin_boundaries) - 1, 
                                           len(self.locs_bin_boundaries) - 1, 
                                           dtype=torch.long),
                       dist_reduce_fx="sum")
        self.add_state("locs_y_cm",
                       default=torch.zeros(len(self.locs_bin_boundaries) - 1, 
                                           len(self.locs_bin_boundaries) - 1, 
                                           dtype=torch.long),
                       dist_reduce_fx="sum")
        self.flux_bands = flux_bands
        self.flux_cms = OrderedDict()
        for flux_b in self.flux_bands:
            s_name = f"{flux_b}_band_flux_cm"
            self.add_state(s_name,
                           default=torch.zeros(len(self.flux_bin_boundaries) - 1, 
                                               len(self.flux_bin_boundaries) - 1, 
                                               dtype=torch.long),
                           dist_reduce_fx="sum")
            self.flux_cms[s_name] = getattr(self, s_name)
        
        font_size = 10
        plt.rcParams.update({
            "text.usetex": False,
            "font.family": "serif",
            "font.size": font_size,
            "axes.labelsize": font_size,
            "axes.titlesize": font_size,
            "legend.fontsize": font_size,
            "xtick.labelsize": font_size,
            "ytick.labelsize": font_size,
            "figure.dpi": 150,
            "lines.linewidth": 2.0,
            "lines.markersize": 8,
        })

    def _refresh_flux_cms(self):
        for k, v in self.flux_cms.items():
            self.flux_cms[k] = getattr(self, k)

    @classmethod
    def _calculate_cm(cls,
                      pred_tensor: torch.Tensor,
                      true_tensor: torch.Tensor,
                      bin_boundaries: torch.Tensor,
                      assert_inclusive: bool,
                      exclude_first_and_last_bins: bool):
        assert pred_tensor.shape == true_tensor.shape
        new_boundary = bin_boundaries.clone()
        new_boundary[0] -= 1e-3
        new_boundary[-1] += 1e-3
        pred_b_index = torch.bucketize(pred_tensor.contiguous(), boundaries=new_boundary)
        true_b_index = torch.bucketize(true_tensor.contiguous(), boundaries=new_boundary)
        bin_num = len(bin_boundaries) + 1
        cm = torch.zeros(bin_num, bin_num, dtype=torch.long, device=pred_tensor.device)
        for ri in range(cm.shape[0]):
            for ci in range(cm.shape[1]):
                cm[ri, ci] = ((pred_b_index == ri) & (true_b_index == ci)).sum()
        if assert_inclusive:
            assert (pred_b_index > 0).all()
            assert (pred_b_index < new_boundary.shape[0]).all()
            assert (true_b_index > 0).all()
            assert (true_b_index < new_boundary.shape[0]).all()
        if exclude_first_and_last_bins:
            cm = cm[1:-1, 1:-1]
        return cm
    
    @classmethod
    def poisson_exact_test(cls, x1, x2):
        """
        Exact two-sided test for equality of two Poisson means.
        """
        n = x1 + x2
        if n == 0:
            return 1.0
        p_lower = stats.binom.cdf(min(x1, x2), n, 0.5)
        p_upper = 1 - stats.binom.cdf(max(x1, x2) - 1, n, 0.5)
        return p_lower + p_upper
    
    @classmethod
    def make_factor_matrix_plot(cls, counts, annot_type, title="", axis_label="", ticks=None):
        p_values = torch.zeros_like(counts, dtype=torch.float32)

        for i in range(len(counts)):
            for j in range(len(counts)):
                if i != j:
                    p_value = cls.poisson_exact_test(counts[i, j], counts[j, i])
                    p_values[i, j] = p_value

        match annot_type:
            case "count":
                annot = counts
                title_postfix = "CM"
                num_fmt = "d"
            case "asymmetry":
                annot = (counts - counts.T) / torch.minimum(counts, counts.T).clamp(min=1)
                title_postfix = "Asymmetry Factor"
                num_fmt = ".2f"
            case "p_values":
                annot = p_values
                title_postfix = "Poisson Exact Test P-value"
                num_fmt = ".2f"
            case _:
                raise NotImplementedError()

        fig, ax = plt.subplots(1, 1, figsize=(2.8, 2.8), constrained_layout=True)
        assert (p_values <= 1.5).all() and (p_values >= 0.0).all()
        sns.heatmap(
            torch.where(p_values > 0.05, 1 / (2 + p_values * 10), 1.0) * (1 - torch.eye(p_values.shape[0])),
            annot=annot,
            fmt=num_fmt,
            cbar=False,
            cmap="YlOrRd",
            vmin=0.0,
            vmax=1.0,
            ax=ax,
        )
        ax.set_xlabel(f"True {axis_label}")
        ax.set_ylabel(f"Pred {axis_label}")
        ax.set_title(f"{title} {title_postfix}")
        ax.tick_params(axis="both", which="major", labelsize=6)
        if ticks is not None:
            ax.set_xticklabels(ticks)
            ax.set_yticklabels(ticks)
        return fig

    def update(self, true_cat, est_cat, matching):
        assert isinstance(true_cat, FullCatalog), "true_cat should be FullCatalog"
        assert isinstance(est_cat, FullCatalog), "est_cat should be FullCatalog"
        assert hasattr(true_cat, "target_tile_cat1")
        assert hasattr(true_cat, "target_tile_cat2")
        assert hasattr(est_cat, "sample_tile_cat")

        self._refresh_flux_cms()

        target_cat1 = true_cat.target_tile_cat1
        target_cat2 = true_cat.target_tile_cat2
        sample_tile_cat = est_cat.sample_tile_cat

        pred_ns = sample_tile_cat["n_sources"]  # (b, h, w)
        true_ns = target_cat1["n_sources"] + target_cat2["n_sources"]
        pred_locs = sample_tile_cat["locs"]  # (b, h, w, 2, 2)
        true_locs = torch.cat([target_cat1["locs"], target_cat2["locs"]], dim=-2)
        pred_flux = sample_tile_cat["fluxes"]  # (b, h, w, 2, k)
        true_flux = torch.cat([target_cat1["fluxes"], target_cat2["fluxes"]], dim=-2)

        ns_cm = torch.zeros(self.max_n_sources + 1, 
                            self.max_n_sources + 1, 
                            dtype=torch.long, 
                            device=pred_ns.device)
        for ri in range(ns_cm.shape[0]):
            for ci in range(ns_cm.shape[1]):
                ns_cm[ri, ci] = ((pred_ns == ri) & (true_ns == ci)).sum()
        self.n_sources_cm += ns_cm
        valid_source_mask = (pred_ns.unsqueeze(-1) >= torch.arange(1, 3, device=pred_ns.device)) & \
                            (pred_ns == true_ns).unsqueeze(-1)  # (b, h, w, 2)
        self.locs_x_cm += self._calculate_cm(pred_locs[valid_source_mask][:, 1], true_locs[valid_source_mask][:, 1],
                                           bin_boundaries=self.locs_bin_boundaries.to(device=pred_ns.device),
                                           assert_inclusive=True, exclude_first_and_last_bins=True)
        self.locs_y_cm += self._calculate_cm(pred_locs[valid_source_mask][:, 0], true_locs[valid_source_mask][:, 0],
                                           bin_boundaries=self.locs_bin_boundaries.to(device=pred_ns.device),
                                           assert_inclusive=True, exclude_first_and_last_bins=True)
        for i, (k, flux_cm) in enumerate(self.flux_cms.items()):
            flux_cm += self._calculate_cm(convert_flux_to_magnitude(pred_flux[valid_source_mask][:, i], zero_point=3631e9), 
                                          convert_flux_to_magnitude(true_flux[valid_source_mask][:, i], zero_point=3631e9),
                                            bin_boundaries=self.flux_bin_boundaries.to(device=pred_ns.device),
                                            assert_inclusive=False, exclude_first_and_last_bins=True)

    def compute(self):
        self._refresh_flux_cms()

        if not _is_global_zero():
            return {}

        fig_dict = {}
        fig_dict["ns_cm_fig"] = self.make_factor_matrix_plot(self.n_sources_cm.cpu(),
                                                            annot_type="count",
                                                            title="N Sources",
                                                            axis_label="Source Count")
        fig_dict["ns_cm_asym_fig"] = self.make_factor_matrix_plot(self.n_sources_cm.cpu(),
                                                            annot_type="asymmetry",
                                                            title="N Sources",
                                                            axis_label="Source Count")
        locs_bin_labels = [f"[{lb.item():.2f}, {rb.item():.2f}]" 
                            for lb, rb in zip(self.locs_bin_boundaries[:-1], 
                                              self.locs_bin_boundaries[1:])]
        fig_dict["locs_x_cm_fig"] = self.make_factor_matrix_plot(self.locs_x_cm.cpu(), 
                                                                annot_type="count", 
                                                                title="Locs X", 
                                                                axis_label="Locs X", 
                                                                ticks=locs_bin_labels)
        fig_dict["locs_x_cm_asym_fig"] = self.make_factor_matrix_plot(self.locs_x_cm.cpu(), 
                                                                    annot_type="asymmetry", 
                                                                    title="Locs X", 
                                                                    axis_label="Locs X", 
                                                                    ticks=locs_bin_labels)
        fig_dict["locs_y_cm_fig"] = self.make_factor_matrix_plot(self.locs_y_cm.cpu(), 
                                                                annot_type="count", 
                                                                title="Locs Y", 
                                                                axis_label="Locs Y", 
                                                                ticks=locs_bin_labels)
        fig_dict["locs_y_cm_asym_fig"] = self.make_factor_matrix_plot(self.locs_y_cm.cpu(), 
                                                                    annot_type="asymmetry", 
                                                                    title="Locs Y", 
                                                                    axis_label="Locs Y", 
                                                                    ticks=locs_bin_labels)
        flux_bin_labels = [f"[{lb.item():.1f}, {rb.item():.1f}]" 
                           for lb, rb in zip(self.flux_bin_boundaries[:-1],
                                             self.flux_bin_boundaries[1:])]
        for k, flux_cm in self.flux_cms.items():
            band = k[0]
            fig_dict[f"{k}_fig"] = self.make_factor_matrix_plot(flux_cm.cpu(), 
                                                                annot_type="count", 
                                                                title=f"{band.upper()}-Band Magnitude",
                                                                axis_label=f"{band.upper()}-Band Magnitude",
                                                                ticks=flux_bin_labels)
            fig_dict[f"{k}_asym_fig"] = self.make_factor_matrix_plot(flux_cm.cpu(), 
                                                                annot_type="asymmetry", 
                                                                title=f"{band.upper()}-Band Magnitude",
                                                                axis_label=f"{band.upper()}-Band Magnitude",
                                                                ticks=flux_bin_labels)
        return fig_dict
