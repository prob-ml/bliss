import torch
from torchmetrics import Metric

from bliss.catalog import FullCatalog
from bliss.encoder.metrics import CatFilter


def _plocs_are_out_boundary(plocs: torch.Tensor, tile_slen: int, boundary_width: float):
    plocs_in_tile = plocs % tile_slen
    return torch.all(
        (plocs_in_tile >= boundary_width) & (plocs_in_tile <= (tile_slen - boundary_width)), dim=-1
    )


def _is_in_boundary(
    plocs: torch.Tensor, is_on_mask: torch.Tensor, tile_slen: int, boundary_width: float
):
    plocs_out_boundary = _plocs_are_out_boundary(
        plocs, tile_slen=tile_slen, boundary_width=boundary_width
    )
    plocs_in_boundary = ~plocs_out_boundary
    plocs_in_boundary &= is_on_mask

    return plocs_in_boundary


def _is_not_in_boundary(
    plocs: torch.Tensor, is_on_mask: torch.Tensor, tile_slen: int, boundary_width: float
):
    plocs_out_boundary = _plocs_are_out_boundary(
        plocs, tile_slen=tile_slen, boundary_width=boundary_width
    )
    plocs_out_boundary &= is_on_mask

    return plocs_out_boundary


class InBoundaryFilter(CatFilter):
    def __init__(self, tile_slen: int, boundary_width: float) -> None:
        super().__init__()
        self.tile_slen = tile_slen
        self.boundary_width = boundary_width

    def get_cur_filter_bools(self, true_cat, est_cat):
        true_filter_bools = _is_in_boundary(
            true_cat["plocs"], true_cat.is_on_mask, self.tile_slen, self.boundary_width
        )
        est_filter_bools = _is_in_boundary(
            est_cat["plocs"], est_cat.is_on_mask, self.tile_slen, self.boundary_width
        )

        return true_filter_bools, est_filter_bools

    def get_cur_postfix(self):
        return "in_boundary"


class OutBoundaryFilter(CatFilter):
    def __init__(self, tile_slen: int, boundary_width: float) -> None:
        super().__init__()
        self.tile_slen = tile_slen
        self.boundary_width = boundary_width

    def get_cur_filter_bools(self, true_cat, est_cat):
        true_filter_bools = _is_not_in_boundary(
            true_cat["plocs"], true_cat.is_on_mask, self.tile_slen, self.boundary_width
        )
        est_filter_bools = _is_not_in_boundary(
            est_cat["plocs"], est_cat.is_on_mask, self.tile_slen, self.boundary_width
        )

        return true_filter_bools, est_filter_bools

    def get_cur_postfix(self):
        return "out_boundary"


class SourceCountFilter(CatFilter):
    def __init__(self, filter_source_count: str) -> None:
        super().__init__()
        self.filter_source_count = filter_source_count
        assert filter_source_count in {
            "1m",
            "2m",
            "2plus",
        }, "invalid filter_source_count"

    def get_cur_filter_bools(self, true_cat, est_cat):
        if self.filter_source_count == "1m":
            true_filter_bools = true_cat["one_source_mask"].squeeze(2)
            est_filter_bools = est_cat["one_source_mask"].squeeze(2)
        elif self.filter_source_count == "2m":
            true_filter_bools = true_cat["two_sources_mask"].squeeze(2)
            est_filter_bools = est_cat["two_sources_mask"].squeeze(2)
        elif self.filter_source_count == "2plus":
            true_filter_bools = true_cat["more_than_two_sources_mask"].squeeze(2)
            est_filter_bools = est_cat["more_than_two_sources_mask"].squeeze(2)
        else:
            raise NotImplementedError()

        return true_filter_bools, est_filter_bools

    def get_cur_postfix(self):
        return self.filter_source_count


class DetectionRecallBlendedness(Metric):
    def __init__(self, bin_cutoffs: list = None) -> None:
        super().__init__()

        self.bin_cutoffs = bin_cutoffs
        assert self.bin_cutoffs, "bin_cutoffs can't be none or empty"

        detection_metrics = [
            "n_true_sources",
            "n_true_matches",
        ]
        for metric in detection_metrics:
            n_bins = len(self.bin_cutoffs) + 1  # fencepost
            init_val = torch.zeros(n_bins)
            self.add_state(metric, default=init_val, dist_reduce_fx="sum")

    def update(self, true_cat, est_cat, matching):
        assert isinstance(true_cat, FullCatalog), "true_cat should be FullCatalog"
        assert isinstance(est_cat, FullCatalog), "est_cat should be FullCatalog"

        true_bin_measures = true_cat["blendedness"].squeeze(-1)

        cutoffs = torch.tensor(self.bin_cutoffs, device=self.device)
        n_bins = len(cutoffs) + 1

        for i in range(true_cat.batch_size):
            tcat_matches, _ = matching[i]
            tcat_matches = tcat_matches.to(device=self.device)
            n_true = true_cat["n_sources"][i].sum().item()
            cur_batch_true_bin_meas = true_bin_measures[i, :n_true]
            true_sources_bin_indices = torch.bucketize(cur_batch_true_bin_meas, cutoffs)

            true_matches_bin_indices = true_sources_bin_indices[tcat_matches.tolist()]

            self.n_true_sources += true_sources_bin_indices.bincount(minlength=n_bins)
            self.n_true_matches += true_matches_bin_indices.bincount(minlength=n_bins)

    def compute(self):
        recall_per_bin = (self.n_true_matches / self.n_true_sources).nan_to_num(0)
        recall = (self.n_true_matches.sum() / self.n_true_sources.sum()).nan_to_num(0)
        recall_bin_results = {
            f"detection_recall_blendedness_bin_{i}": recall_per_bin[i]
            for i in range(len(recall_per_bin))
        }

        return {
            "detection_recall_blendedness": recall,
            **recall_bin_results,
        }

    def get_results_on_per_bin(self):
        recall = (self.n_true_matches / self.n_true_sources).nan_to_num(0)
        return {
            "detection_recall_blendedness": recall,
        }
