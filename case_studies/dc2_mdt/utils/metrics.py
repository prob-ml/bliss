import torch

from typing import List
from torchmetrics import Metric
from einops import rearrange

from bliss.catalog import FullCatalog
from bliss.encoder.metrics import CatFilter, NullFilter

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
