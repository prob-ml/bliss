import torch
from torchmetrics import Metric

from bliss.catalog import FullCatalog
from bliss.encoder.metrics import CatFilter, FilterMetric


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


class Cosmodc2Filter(CatFilter):
    def get_cur_filter_bools(self, true_cat, est_cat):
        true_filter_bools = true_cat["cosmodc2_mask"].squeeze(2)
        return true_filter_bools, None

    def get_cur_postfix(self):
        return "cosmodc2"


class ShearMSE(FilterMetric):
    def __init__(
        self,
        bin_cutoffs: list,
        ref_band: int = 2,
        bin_type: str = "nmgy",
    ):
        # shear is only for cosmodc2
        super().__init__([Cosmodc2Filter()])

        self.bin_cutoffs = bin_cutoffs
        self.ref_band = ref_band
        self.bin_type = bin_type

        assert self.bin_cutoffs, "flux_bin_cutoffs can't be None or empty"
        assert self.bin_type in {"nmgy", "njymag"}, "invalid bin type"

        n_bins = len(self.bin_cutoffs) + 1

        # sum squared error (sse)
        self.add_state("shear1_sse", default=torch.zeros(n_bins), dist_reduce_fx="sum")
        self.add_state("shear2_sse", default=torch.zeros(n_bins), dist_reduce_fx="sum")
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

            shear1_se = (
                true_cat["shear"][i, tcat_matches, 0] - est_cat["shear"][i, ecat_matches, 0]
            ) ** 2
            shear2_se = (
                true_cat["shear"][i, tcat_matches, 1] - est_cat["shear"][i, ecat_matches, 1]
            ) ** 2
            shear1_se = torch.split(shear1_se[to_bin_mapping], per_bin_elements_count.tolist())
            shear2_se = torch.split(shear2_se[to_bin_mapping], per_bin_elements_count.tolist())

            shear1_sse = torch.tensor([i.sum() for i in shear1_se], device=self.device)
            shear2_sse = torch.tensor([i.sum() for i in shear2_se], device=self.device)

            self.shear1_sse += shear1_sse
            self.shear2_sse += shear2_sse
            self.n_matches += per_bin_elements_count

    def compute(self):
        shear1_mse = (self.shear1_sse.sum() / self.n_matches.sum()).nan_to_num(0)
        shear2_mae = (self.shear2_sse.sum() / self.n_matches.sum()).nan_to_num(0)
        shear1_mse_per_bin = (self.shear1_sse / self.n_matches).nan_to_num(0)
        shear2_mse_per_bin = (self.shear2_sse / self.n_matches).nan_to_num(0)

        shear1_mse_per_bin_results = {
            f"shear1_mse_bin_{i}": shear1_mse_per_bin[i] for i in range(len(shear1_mse_per_bin))
        }
        shear2_mse_per_bin_results = {
            f"shear2_mse_bin_{i}": shear2_mse_per_bin[i] for i in range(len(shear2_mse_per_bin))
        }

        return {
            "shear1_mse": shear1_mse,
            "shear2_mse": shear2_mae,
            **shear1_mse_per_bin_results,
            **shear2_mse_per_bin_results,
        }


class EllipticityMSE(FilterMetric):
    def __init__(
        self,
        bin_cutoffs: list,
        ref_band: int = 2,
        bin_type: str = "nmgy",
    ):
        # ellipticity is only for cosmodc2
        super().__init__([Cosmodc2Filter()])

        self.bin_cutoffs = bin_cutoffs
        self.ref_band = ref_band
        self.bin_type = bin_type

        assert self.bin_cutoffs, "flux_bin_cutoffs can't be None or empty"
        assert self.bin_type in {"nmgy", "njymag", "blendedness"}, "invalid bin type"

        n_bins = len(self.bin_cutoffs) + 1

        # sum squared error (sse)
        self.add_state("g1_sse", default=torch.zeros(n_bins), dist_reduce_fx="sum")
        self.add_state("g2_sse", default=torch.zeros(n_bins), dist_reduce_fx="sum")
        self.add_state("n_matches", default=torch.zeros(n_bins), dist_reduce_fx="sum")

    def update(self, true_cat, est_cat, matching):
        cutoffs = torch.tensor(self.bin_cutoffs, device=self.device)
        n_bins = len(cutoffs) + 1

        if self.bin_type in {"nmgy", "mag", "njymag"}:
            true_bin_measures = true_cat.on_fluxes(self.bin_type)[:, :, self.ref_band].contiguous()
        elif self.bin_type == "blendedness":
            true_bin_measures = true_cat["blendedness"].squeeze(-1).contiguous()
        else:
            raise NotImplementedError()

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

            g1_se = (
                true_cat["ellipticity"][i, tcat_matches, 0]
                - est_cat["ellipticity"][i, ecat_matches, 0]
            ) ** 2
            g2_se = (
                true_cat["ellipticity"][i, tcat_matches, 1]
                - est_cat["ellipticity"][i, ecat_matches, 1]
            ) ** 2
            g1_se = torch.split(g1_se[to_bin_mapping], per_bin_elements_count.tolist())
            g2_se = torch.split(g2_se[to_bin_mapping], per_bin_elements_count.tolist())

            g1_sse = torch.tensor([i.sum() for i in g1_se], device=self.device)
            g2_sse = torch.tensor([i.sum() for i in g2_se], device=self.device)

            self.g1_sse += g1_sse
            self.g2_sse += g2_sse
            self.n_matches += per_bin_elements_count

    def compute(self):
        g1_mse = (self.g1_sse.sum() / self.n_matches.sum()).nan_to_num(0)
        g2_mse = (self.g2_sse.sum() / self.n_matches.sum()).nan_to_num(0)
        g1_mse_per_bin = (self.g1_sse / self.n_matches).nan_to_num(0)
        g2_mse_per_bin = (self.g2_sse / self.n_matches).nan_to_num(0)

        g1_mse_per_bin_results = {
            f"g1_mse_bin_{i}": g1_mse_per_bin[i] for i in range(len(g1_mse_per_bin))
        }
        g2_mse_per_bin_results = {
            f"g2_mse_bin_{i}": g2_mse_per_bin[i] for i in range(len(g2_mse_per_bin))
        }

        return {
            "g1_mse": g1_mse,
            "g2_mse": g2_mse,
            **g1_mse_per_bin_results,
            **g2_mse_per_bin_results,
        }

    def get_results_on_per_bin(self):
        g1_mse_per_bin = (self.g1_sse / self.n_matches).nan_to_num(0)
        g2_mse_per_bin = (self.g2_sse / self.n_matches).nan_to_num(0)

        return {
            "g1_mse": g1_mse_per_bin,
            "g2_mse": g2_mse_per_bin,
        }


class EllipticityResidual(FilterMetric):
    def __init__(
        self,
        bin_cutoffs: list,
        ref_band: int = 2,
        bin_type: str = "nmgy",
    ):
        # ellipticity is only for cosmodc2
        super().__init__([Cosmodc2Filter()])

        self.bin_cutoffs = bin_cutoffs
        self.ref_band = ref_band
        self.bin_type = bin_type

        assert self.bin_cutoffs, "flux_bin_cutoffs can't be None or empty"
        assert self.bin_type in {"nmgy", "njymag", "blendedness"}, "invalid bin type"

        n_bins = len(self.bin_cutoffs) + 1

        self.add_state("g1_residual", default=torch.zeros(n_bins), dist_reduce_fx="sum")
        self.add_state("g2_residual", default=torch.zeros(n_bins), dist_reduce_fx="sum")
        self.add_state("n_matches", default=torch.zeros(n_bins), dist_reduce_fx="sum")

    def update(self, true_cat, est_cat, matching):
        cutoffs = torch.tensor(self.bin_cutoffs, device=self.device)
        n_bins = len(cutoffs) + 1

        if self.bin_type in {"nmgy", "mag", "njymag"}:
            true_bin_measures = true_cat.on_fluxes(self.bin_type)[:, :, self.ref_band].contiguous()
        elif self.bin_type == "blendedness":
            true_bin_measures = true_cat["blendedness"].squeeze(-1).contiguous()
        else:
            raise NotImplementedError()

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

            g1_residual = (
                est_cat["ellipticity"][i, ecat_matches, 0]
                - true_cat["ellipticity"][i, tcat_matches, 0]
            )
            g2_residual = (
                est_cat["ellipticity"][i, ecat_matches, 1]
                - true_cat["ellipticity"][i, tcat_matches, 1]
            )
            g1_residual = torch.split(g1_residual[to_bin_mapping], per_bin_elements_count.tolist())
            g2_residual = torch.split(g2_residual[to_bin_mapping], per_bin_elements_count.tolist())

            g1_sum_residual = torch.tensor([i.sum() for i in g1_residual], device=self.device)
            g2_sum_residual = torch.tensor([i.sum() for i in g2_residual], device=self.device)

            self.g1_residual += g1_sum_residual
            self.g2_residual += g2_sum_residual
            self.n_matches += per_bin_elements_count

    def compute(self):
        g1_residual = (self.g1_residual.sum() / self.n_matches.sum()).nan_to_num(0)
        g2_residual = (self.g2_residual.sum() / self.n_matches.sum()).nan_to_num(0)
        g1_residual_per_bin = (self.g1_residual / self.n_matches).nan_to_num(0)
        g2_residual_per_bin = (self.g2_residual / self.n_matches).nan_to_num(0)

        g1_residual_per_bin_results = {
            f"g1_residual_bin_{i}": g1_residual_per_bin[i] for i in range(len(g1_residual_per_bin))
        }
        g2_residual_per_bin_results = {
            f"g2_residual_bin_{i}": g2_residual_per_bin[i] for i in range(len(g2_residual_per_bin))
        }

        return {
            "g1_residual": g1_residual,
            "g2_residual": g2_residual,
            **g1_residual_per_bin_results,
            **g2_residual_per_bin_results,
        }

    def get_results_on_per_bin(self):
        g1_residual_per_bin = (self.g1_residual / self.n_matches).nan_to_num(0)
        g2_residual_per_bin = (self.g2_residual / self.n_matches).nan_to_num(0)

        return {
            "g1_residual": g1_residual_per_bin,
            "g2_residual": g2_residual_per_bin,
        }
