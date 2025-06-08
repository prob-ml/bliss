# pylint: disable=R0801
from typing import List

import torch

from bliss.catalog import FullCatalog
from bliss.encoder.metrics import CatFilter, FluxBinMetricWithFilter, GeneralBinMetric


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


class DetectionRecallwrtBlendedness(GeneralBinMetric):
    def __init__(self, bin_cutoffs: list, exclude_last_bin: bool = False) -> None:
        super().__init__(bin_cutoffs, exclude_last_bin)
        assert self.bin_cutoffs, "bin_cutoffs can't be none or empty"

        self.add_bin_state("n_true_sources")
        self.add_bin_state("n_true_matches")

    def update(self, true_cat, est_cat, matching):
        assert isinstance(true_cat, FullCatalog), "true_cat should be FullCatalog"
        assert isinstance(est_cat, FullCatalog), "est_cat should be FullCatalog"

        true_blendedness = true_cat["blendedness"].squeeze(-1)

        for i in range(true_cat.batch_size):
            tcat_matches, _ = matching[i]
            tcat_matches = tcat_matches.to(device=self.device)
            n_true = true_cat["n_sources"][i].sum().item()
            cur_batch_true_blendedness = true_blendedness[i, :n_true]
            true_sources_bin_indexes = self.bucketize(cur_batch_true_blendedness)

            true_matches_bin_indices = true_sources_bin_indexes[tcat_matches]

            self.n_true_sources += self.bincount(true_sources_bin_indexes)
            self.n_true_matches += self.bincount(true_matches_bin_indices)

    def compute(self):
        n_true_matches = self.get_state_for_report("n_true_matches")
        n_true_sources = self.get_state_for_report("n_true_sources")

        recall_per_bin = (n_true_matches / n_true_sources).nan_to_num(0)
        recall = (n_true_matches.sum() / n_true_sources.sum()).nan_to_num(0)
        recall_bin_results = {
            f"detection_recall_blendedness_bin_{i}": bin_recall
            for i, bin_recall in enumerate(recall_per_bin)
        }

        return {
            "detection_recall_blendedness": recall,
            **recall_bin_results,
        }

    def get_results_on_per_bin(self):
        n_true_matches = self.get_state_for_report("n_true_matches")
        n_true_sources = self.get_state_for_report("n_true_sources")

        recall = (n_true_matches / n_true_sources).nan_to_num(0)
        return {
            "detection_recall_blendedness": recall,
        }

    def get_internal_states(self):
        n_true_matches = self.get_state_for_report("n_true_matches")
        n_true_sources = self.get_state_for_report("n_true_sources")

        return {
            "n_true_sources": n_true_sources.nan_to_num(0),
            "n_true_matches": n_true_matches.nan_to_num(0),
        }


class Cosmodc2Filter(CatFilter):
    def get_cur_filter_bools(self, true_cat, est_cat):
        true_filter_bools = true_cat["cosmodc2_mask"].squeeze(2)
        return true_filter_bools, None

    def get_cur_postfix(self):
        return "cosmodc2"


class ShearMSE(FluxBinMetricWithFilter):
    def __init__(
        self,
        base_flux_bin_cutoffs: list,
        mag_zero_point: int,
        ref_band: int = 2,
        report_bin_unit: str = "mag",
        exclude_last_bin: bool = False,
    ):
        # shear is only for cosmodc2
        super().__init__(
            base_flux_bin_cutoffs,
            mag_zero_point,
            report_bin_unit,
            exclude_last_bin,
            [Cosmodc2Filter()],
        )
        assert self.base_flux_bin_cutoffs, "cutoffs can't be None or empty"

        self.ref_band = ref_band

        # sum squared error (sse)
        self.add_bin_state("shear1_sse")
        self.add_bin_state("shear2_sse")
        self.add_bin_state("n_matches")

    def update(self, true_cat, est_cat, matching):
        true_fluxes = true_cat.on_fluxes[:, :, self.ref_band].contiguous()

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

            cur_batch_true_fluxes = true_fluxes[i][tcat_matches]
            bin_indexes = self.bucketize(cur_batch_true_fluxes)
            self.n_matches += self.bincount(bin_indexes)

            shear1_se = (
                true_cat["shear"][i, tcat_matches, 0] - est_cat["shear"][i, ecat_matches, 0]
            ) ** 2
            shear2_se = (
                true_cat["shear"][i, tcat_matches, 1] - est_cat["shear"][i, ecat_matches, 1]
            ) ** 2
            tmp = torch.zeros(self.n_bins, dtype=torch.float, device=self.device)
            self.shear1_sse += tmp.scatter_add(dim=0, index=bin_indexes, src=shear1_se)
            self.shear2_sse += tmp.scatter_add(dim=0, index=bin_indexes, src=shear2_se)

    def compute(self):
        shear1_sse = self.get_state_for_report("shear1_sse")
        shear2_sse = self.get_state_for_report("shear2_sse")
        n_matches = self.get_state_for_report("n_matches")

        shear1_mse = (shear1_sse.sum() / n_matches.sum()).nan_to_num(0)
        shear2_mse = (shear2_sse.sum() / n_matches.sum()).nan_to_num(0)
        shear1_mse_per_bin = (shear1_sse / n_matches).nan_to_num(0)
        shear2_mse_per_bin = (shear2_sse / n_matches).nan_to_num(0)

        shear1_mse_per_bin_results = {
            f"shear1_mse_bin_{i}": bin_shear1_mse
            for i, bin_shear1_mse in enumerate(shear1_mse_per_bin)
        }
        shear2_mse_per_bin_results = {
            f"shear2_mse_bin_{i}": bin_shear2_mse
            for i, bin_shear2_mse in enumerate(shear2_mse_per_bin)
        }

        return {
            "shear1_mse": shear1_mse,
            "shear2_mse": shear2_mse,
            **shear1_mse_per_bin_results,
            **shear2_mse_per_bin_results,
        }


class EllipticityMSE(FluxBinMetricWithFilter):
    def __init__(
        self,
        base_flux_bin_cutoffs: list,
        mag_zero_point: int,
        ref_band: int = 2,
        report_bin_unit: str = "mag",
        exclude_last_bin: bool = False,
    ):
        # shear is only for cosmodc2
        super().__init__(
            base_flux_bin_cutoffs,
            mag_zero_point,
            report_bin_unit,
            exclude_last_bin,
            [Cosmodc2Filter()],
        )
        assert self.base_flux_bin_cutoffs, "cutoffs can't be None or empty"

        self.ref_band = ref_band

        # sum squared error (sse)
        self.add_bin_state("g1_sse")
        self.add_bin_state("g2_sse")
        self.add_bin_state("n_matches")

    def update(self, true_cat, est_cat, matching):
        true_fluxes = true_cat.on_fluxes[:, :, self.ref_band].contiguous()

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

            cur_batch_true_fluxes = true_fluxes[i][tcat_matches]
            bin_indexes = self.bucketize(cur_batch_true_fluxes)
            self.n_matches += self.bincount(bin_indexes)

            g1_se = (
                true_cat["ellipticity"][i, tcat_matches, 0]
                - est_cat["ellipticity"][i, ecat_matches, 0]
            ) ** 2
            g2_se = (
                true_cat["ellipticity"][i, tcat_matches, 1]
                - est_cat["ellipticity"][i, ecat_matches, 1]
            ) ** 2
            tmp_zeros = torch.zeros(self.n_bins, dtype=torch.float, device=self.device)
            self.g1_sse += tmp_zeros.scatter_add(dim=0, index=bin_indexes, src=g1_se)
            self.g2_sse += tmp_zeros.scatter_add(dim=0, index=bin_indexes, src=g2_se)

    def compute(self):
        g1_sse = self.get_state_for_report("g1_sse")
        g2_sse = self.get_state_for_report("g2_sse")
        n_matches = self.get_state_for_report("n_matches")

        g1_mse = (g1_sse.sum() / n_matches.sum()).nan_to_num(0)
        g2_mse = (g2_sse.sum() / n_matches.sum()).nan_to_num(0)
        g1_mse_per_bin = (g1_sse / n_matches).nan_to_num(0)
        g2_mse_per_bin = (g2_sse / n_matches).nan_to_num(0)

        g1_mse_per_bin_results = {
            f"g1_mse_bin_{i}": bin_g1_mse for i, bin_g1_mse in enumerate(g1_mse_per_bin)
        }
        g2_mse_per_bin_results = {
            f"g2_mse_bin_{i}": bin_g2_mse for i, bin_g2_mse in enumerate(g2_mse_per_bin)
        }

        return {
            "g1_mse": g1_mse,
            "g2_mse": g2_mse,
            **g1_mse_per_bin_results,
            **g2_mse_per_bin_results,
        }

    def get_results_on_per_bin(self):
        g1_sse = self.get_state_for_report("g1_sse")
        g2_sse = self.get_state_for_report("g2_sse")
        n_matches = self.get_state_for_report("n_matches")

        g1_mse_per_bin = (g1_sse / n_matches).nan_to_num(0)
        g2_mse_per_bin = (g2_sse / n_matches).nan_to_num(0)

        return {
            "g1_mse": g1_mse_per_bin,
            "g2_mse": g2_mse_per_bin,
        }


class EllipticityResidual(FluxBinMetricWithFilter):
    def __init__(
        self,
        base_flux_bin_cutoffs: list,
        mag_zero_point: int,
        ref_band: int = 2,
        report_bin_unit: str = "mag",
        exclude_last_bin: bool = False,
    ):
        # ellipticity is only for cosmodc2
        super().__init__(
            base_flux_bin_cutoffs,
            mag_zero_point,
            report_bin_unit,
            exclude_last_bin,
            [Cosmodc2Filter()],
        )
        assert self.base_flux_bin_cutoffs, "cutoffs can't be None or empty"

        self.ref_band = ref_band

        self.add_bin_state("g1_residual")
        self.add_bin_state("g2_residual")

    def update(self, true_cat, est_cat, matching):
        true_fluxes = true_cat.on_fluxes[:, :, self.ref_band].contiguous()

        true_filter_bools, _ = self.get_filter_bools(true_cat, est_cat)

        self.g1_residual.zero_()
        self.g2_residual.zero_()
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
            bin_indexes = self.bucketize(cur_batch_true_fluxes)

            g1_residual = (
                est_cat["ellipticity"][i, ecat_matches, 0]
                - true_cat["ellipticity"][i, tcat_matches, 0]
            )
            g2_residual = (
                est_cat["ellipticity"][i, ecat_matches, 1]
                - true_cat["ellipticity"][i, tcat_matches, 1]
            )

            # equivalent to randomly choose a residual
            self.g1_residual.scatter_(dim=0, index=bin_indexes, src=g1_residual)
            self.g2_residual.scatter_(dim=0, index=bin_indexes, src=g2_residual)

    def compute(self):
        # this class is not for training or validation
        raise NotImplementedError()

    def get_results_on_per_bin(self):
        g1_residual = self.get_state_for_report("g1_residual")
        g2_residual = self.get_state_for_report("g2_residual")

        return {
            "g1_residual": g1_residual.nan_to_num(0),
            "g2_residual": g2_residual.nan_to_num(0),
        }


class GeneralBinMetricWithFilter(GeneralBinMetric):
    def __init__(self, bin_cutoffs: list, exclude_last_bin: bool, filter_list: List[CatFilter]):
        super().__init__(bin_cutoffs, exclude_last_bin)

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
            return f"_{'_'.join(postfix_list)}"

        return ""


class EllipticityResidualwrtBlendedness(GeneralBinMetricWithFilter):
    def __init__(
        self,
        bin_cutoffs: list,
        ref_band: int = 2,
        exclude_last_bin: bool = False,
    ):
        # ellipticity is only for cosmodc2
        super().__init__(bin_cutoffs, exclude_last_bin, [Cosmodc2Filter()])
        assert self.bin_cutoffs, "cutoffs can't be None or empty"

        self.ref_band = ref_band

        self.add_bin_state("g1_residual")
        self.add_bin_state("g2_residual")

    def update(self, true_cat, est_cat, matching):
        true_blendedness = true_cat["blendedness"].squeeze(-1)

        true_filter_bools, _ = self.get_filter_bools(true_cat, est_cat)

        self.g1_residual.zero_()
        self.g2_residual.zero_()
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

            cur_batch_true_blendedness = true_blendedness[i][tcat_matches]
            bin_indexes = self.bucketize(cur_batch_true_blendedness)

            g1_residual = (
                est_cat["ellipticity"][i, ecat_matches, 0]
                - true_cat["ellipticity"][i, tcat_matches, 0]
            )
            g2_residual = (
                est_cat["ellipticity"][i, ecat_matches, 1]
                - true_cat["ellipticity"][i, tcat_matches, 1]
            )

            # equivalent to randomly choose a residual
            self.g1_residual.scatter_(dim=0, index=bin_indexes, src=g1_residual)
            self.g2_residual.scatter_(dim=0, index=bin_indexes, src=g2_residual)

    def compute(self):
        # this class is not for training or validation
        raise NotImplementedError()

    def get_results_on_per_bin(self):
        g1_residual = self.get_state_for_report("g1_residual")
        g2_residual = self.get_state_for_report("g2_residual")

        return {
            "g1_residual": g1_residual.nan_to_num(0),
            "g2_residual": g2_residual.nan_to_num(0),
        }
