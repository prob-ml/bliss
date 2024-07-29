import copy
from abc import ABC, abstractmethod
from typing import List

import numpy as np
import torch
from einops import rearrange
from scipy.optimize import linear_sum_assignment

from bliss.catalog import TileCatalog
from bliss.encoder.variational_dist import VariationalFactor
from bliss.surveys.dc2 import unpack_dict


class Assigner(ABC):
    @abstractmethod
    def __call__(self, est_tile_cat: TileCatalog, true_tile_cat: TileCatalog, topk: int):
        """Get est_to_true_indices of shape (B, m, 1)."""


class DistAssigner(Assigner):
    @classmethod
    def _get_dist_matrix(cls, est_tile_cat, true_tile_cat):
        est_locs = est_tile_cat["locs"]
        true_locs = true_tile_cat["locs"]
        est_is_on_mask = est_tile_cat["n_sources_mask"]  # (b, nth, ntw, m, 1)
        true_is_on_mask = rearrange(true_tile_cat.is_on_mask, "b nth ntw m -> b nth ntw m 1")
        est_locs = torch.where(est_is_on_mask, est_locs, torch.full_like(est_locs, 1e2))
        true_locs = torch.where(true_is_on_mask, true_locs, torch.full_like(true_locs, 1e2))
        est_locs = rearrange(est_locs, "b nth ntw m k -> (b nth ntw) m k")
        true_locs = rearrange(true_locs, "b nth ntw m k -> (b nth ntw) m k")
        return torch.cdist(est_locs, true_locs, p=2)  # (B, m, m)


class PartialDistAssigner(Assigner):
    @classmethod
    def _get_partial_dist_matrix(cls, est_tile_cat, true_tile_cat):
        est_locs = est_tile_cat["locs"]
        true_locs = true_tile_cat["locs"]
        est_locs = rearrange(est_locs, "b nth ntw m k -> (b nth ntw) m k")
        true_locs = rearrange(true_locs, "b nth ntw m k -> (b nth ntw) m k")
        true_is_on_mask = rearrange(true_tile_cat.is_on_mask, "b nth ntw m -> (b nth ntw) 1 m")
        return torch.cdist(est_locs, true_locs, p=2), true_is_on_mask


class FastOne2OneAssigner(DistAssigner):
    @classmethod
    def _fast_one_to_one_match(
        cls, dist_matrix: torch.Tensor, est_is_on_mask: torch.Tensor, true_is_on_mask: torch.Tensor
    ):
        assert dist_matrix.shape[1] == dist_matrix.shape[2]
        m = dist_matrix.shape[1]
        _, match1 = torch.min(dist_matrix, dim=-2)  # (B, m)
        _, match2 = torch.min(dist_matrix, dim=-1)  # (B, m)
        matches = (match1, match2)
        est_to_true_indices = torch.stack(matches, dim=-1)  # (B, m, 2) 2 represents index

        # find one-to-one matching
        match1cond2 = torch.gather(match1, 1, match2)  # (B, m)
        pos_tensor = torch.arange(
            0, m, dtype=est_to_true_indices.dtype, device=est_to_true_indices.device
        ).view(
            1, -1
        )  # (1, m)
        one_to_one_match = (match1cond2 == pos_tensor).unsqueeze(-1)  # (B, m, 1)
        est_on_mask_num = rearrange(est_is_on_mask, "b nth ntw m 1 -> (b nth ntw) m").sum(
            dim=-1, keepdim=True
        )  # (B, 1)
        true_on_mask_num = rearrange(true_is_on_mask, "b nth ntw m 1 -> (b nth ntw) m").sum(
            dim=-1, keepdim=True
        )
        est_out_mask = est_to_true_indices[..., 0] > (est_on_mask_num - 1)  # (B, m)
        true_out_mask = est_to_true_indices[..., 1] > (true_on_mask_num - 1)  # (B, m)
        one_to_one_match &= ~(est_out_mask | true_out_mask).unsqueeze(-1)

        offset = torch.where(one_to_one_match, 0, m)  # (B, m, 1)
        est_to_true_indices = est_to_true_indices * one_to_one_match + offset
        pad = torch.zeros_like(est_to_true_indices)[:, 0:1, :] + m
        est_to_true_indices = torch.cat((est_to_true_indices, pad), dim=1)  # (B, m + 1, 2)
        return torch.scatter(
            torch.zeros_like(est_to_true_indices[..., 0]) + m,
            dim=-1,
            index=est_to_true_indices[..., 0],
            src=est_to_true_indices[..., 1],
        )[:, :-1].unsqueeze(
            -1
        )  # (B, m, 1)

    def __call__(self, est_tile_cat: TileCatalog, true_tile_cat: TileCatalog, topk: int):
        locs_distance = self._get_dist_matrix(est_tile_cat, true_tile_cat)
        est_is_on_mask = rearrange(est_tile_cat.is_on_mask, "b nth ntw m -> b nth ntw m 1")
        true_is_on_mask = rearrange(true_tile_cat.is_on_mask, "b nth ntw m -> b nth ntw m 1")
        return self._fast_one_to_one_match(locs_distance, est_is_on_mask, true_is_on_mask)


class LinearSumAssigner(DistAssigner):
    @classmethod
    def _linear_sum_assignment(cls, dist_matrix: torch.Tensor):
        ori_device = dist_matrix.device
        dist_matrix_np = dist_matrix.cpu().numpy()
        est_to_true_indices_list = []
        for i in range(dist_matrix.shape[0]):
            row_index, col_index = linear_sum_assignment(dist_matrix_np[i])
            cur_indices = np.stack((row_index, col_index), axis=-1)
            est_to_true_indices_list.append(cur_indices)
        est_to_true_indices = np.stack(est_to_true_indices_list, axis=0)
        est_to_true_indices = torch.from_numpy(est_to_true_indices).to(device=ori_device)
        est_to_true_indices = torch.scatter(
            torch.zeros_like(est_to_true_indices[..., 0]),
            dim=-1,
            index=est_to_true_indices[..., 0],
            src=est_to_true_indices[..., 1],
        )
        return est_to_true_indices.unsqueeze(-1)

    def __call__(self, est_tile_cat: TileCatalog, true_tile_cat: TileCatalog, topk: int):
        locs_distance = self._get_dist_matrix(est_tile_cat, true_tile_cat)
        return self._linear_sum_assignment(locs_distance)


class TaskAlignedAssigner(PartialDistAssigner):
    @classmethod
    def _get_est_to_true_mapping(cls, est_tile_cat, true_tile_cat, topk):
        est_to_true_locs_dist, true_mask = cls._get_partial_dist_matrix(
            est_tile_cat, true_tile_cat
        )  # (B, m, m), (B, 1, m)
        true_to_est_locs_dist, true_mask = est_to_true_locs_dist.permute(
            0, 2, 1
        ), true_mask.permute(
            0, 2, 1
        )  # (B, m, m), (B, m, 1)
        assert topk <= true_to_est_locs_dist.shape[-1]
        _, true_to_est_topk_indices = torch.topk(
            true_to_est_locs_dist, k=topk, dim=-1, largest=False
        )  # (B, m, topk)

        # (B, m, topk) -> (B, m, m)
        # create a blank mapping
        true_to_est_mapping = torch.zeros(
            *true_to_est_locs_dist.shape,
            dtype=torch.bool,
            device=true_to_est_locs_dist.device,
        )
        true_to_est_mapping.scatter_(dim=-1, index=true_to_est_topk_indices, value=True)
        true_to_est_mapping &= true_mask

        est_mask = (true_to_est_mapping.sum(dim=-2) > 0).unsqueeze(-1)  # (B, m, 1)
        if true_to_est_mapping.sum(dim=-2).max() > 1:
            dist_matrix = torch.where(
                true_to_est_mapping, true_to_est_locs_dist, torch.inf
            )  # (B, m, m)
            one_est_indices = dist_matrix.argmin(dim=-2).unsqueeze(-1)  # (B, m, 1)
            est_to_true_mapping = torch.zeros(
                *est_to_true_locs_dist.shape,
                dtype=torch.bool,
                device=est_to_true_locs_dist.device,
            )
            est_to_true_mapping.scatter_(dim=-1, index=one_est_indices, value=True)  # (B, m, m)
            est_to_true_mapping &= est_mask
            est_to_true_mapping &= true_mask.permute(0, 2, 1)
        else:
            est_to_true_mapping = true_to_est_mapping.permute(0, 2, 1)

        return est_to_true_mapping, est_mask

    def __call__(self, est_tile_cat: TileCatalog, true_tile_cat: TileCatalog, topk: int):
        assert topk >= 1
        est_to_true_mapping, est_mask = self._get_est_to_true_mapping(
            est_tile_cat, true_tile_cat, topk
        )  # (B, m, m), (B, m, 1)
        assert (est_to_true_mapping.sum(dim=-1) <= 1).all()
        est_to_true_indices = est_to_true_mapping.to(dtype=torch.int8).argmax(
            dim=-1, keepdim=True
        )  # (B, m, 1)
        return torch.where(est_mask, est_to_true_indices, est_tile_cat.max_sources)


class MultiVariationalDist(torch.nn.Module):
    def __init__(
        self,
        tile_slen: int,
        factors: List[VariationalFactor],
        repeat_times: int,
        assigner: Assigner,
    ):
        super().__init__()

        self.tile_slen = tile_slen
        self.factors = factors
        self.repeat_times = repeat_times
        self.split_sizes = [v.n_params for v in self.factors]
        assert repeat_times > 1
        self.assigner = assigner

    def _separate_x_cat(self, x_cat):
        chunk_size = sum(self.split_sizes)
        return torch.split(x_cat, chunk_size, 3)

    @classmethod
    def _stack_tile_cat(cls, tile_cat_list: List[TileCatalog]):
        output_tile_cat = None
        for tile_cat in tile_cat_list:
            if output_tile_cat is None:
                output_tile_cat = tile_cat
            else:
                output_tile_cat = output_tile_cat.stack(tile_cat)
        return output_tile_cat

    @classmethod
    def _pad_along_max_sources(cls, v, pad_value=0):
        """Pad (b, nth, ntw, m, k) to be (b, nth, ntw, m + 1, k)."""
        pad = torch.zeros_like(v[:, :, :, 0:1, :])
        if pad_value != 0:
            pad += pad_value
        return torch.cat((v, pad), dim=-2)

    @torch.no_grad()
    def _match_tile_cat(
        self,
        est_tile_cat: TileCatalog,
        true_tile_cat: TileCatalog,
        topk: int,
    ):
        assert est_tile_cat.max_sources == true_tile_cat.max_sources
        est_tile_dict = copy.copy(est_tile_cat.data)
        true_tile_dict = copy.copy(true_tile_cat.data)
        b, nth, ntw, m = est_tile_dict["locs"].shape[:-1]

        est_to_true_indices = self.assigner(est_tile_cat, true_tile_cat, topk)
        est_to_true_indices = rearrange(
            est_to_true_indices, "(b nth ntw) m 1 -> b nth ntw m 1", b=b, nth=nth, ntw=ntw
        )
        # to be (b, nth, ntw, m + 1, 1)
        est_to_true_indices = self._pad_along_max_sources(est_to_true_indices, pad_value=m)

        padded_true_tile_dict = {
            k: self._pad_along_max_sources(v)
            for k, v in true_tile_dict.items()
            if k != "n_sources" and k in est_tile_dict
        }
        true_is_on_mask = rearrange(true_tile_cat.is_on_mask, "b nth ntw m -> b nth ntw m 1")
        padded_true_tile_dict["n_sources"] = self._pad_along_max_sources(true_is_on_mask).to(
            dtype=true_tile_dict["n_sources"].dtype
        )
        cosmodc2_mask = true_tile_dict.get("cosmodc2_mask", None)
        if cosmodc2_mask is not None:
            padded_true_tile_dict["cosmodc2_mask"] = self._pad_along_max_sources(cosmodc2_mask)
        target_tile_dict = {k: torch.zeros_like(v) for k, v in padded_true_tile_dict.items()}

        for k, v in target_tile_dict.items():
            true_v = padded_true_tile_dict[k]  # (b, nth, ntw, m + 1, k)
            expanded_est_to_true_indices = est_to_true_indices.expand_as(true_v)
            v = torch.gather(true_v, dim=-2, index=expanded_est_to_true_indices)
            target_tile_dict[k] = v[:, :, :, :-1, :].split(1, dim=-2)

        # squeeze n_sources
        target_tile_dict["n_sources"] = [
            rearrange(v, "b nth ntw 1 1 -> b nth ntw") for v in target_tile_dict["n_sources"]
        ]

        # get output
        target_tile_dict_list = unpack_dict(target_tile_dict)
        return [TileCatalog(d) for d in target_tile_dict_list]

    @property
    def n_params_per_source(self):
        return sum(self.split_sizes) * self.repeat_times

    def _factor_param_pairs(self, x_cat):
        separated_x_cat = self._separate_x_cat(x_cat)
        dist_params_nested_list = []
        for sub_x_cat in separated_x_cat:
            dist_params_nested_list.append(torch.split(sub_x_cat, self.split_sizes, 3))
        return [zip(self.factors, dist_params_list) for dist_params_list in dist_params_nested_list]

    def _individual_sample(self, x_cat, use_mode=False):
        fp_pairs_list = self._factor_param_pairs(x_cat)
        output_tile_cat_list = []
        for fp_pairs in fp_pairs_list:
            d = {qk.name: qk.sample(params, use_mode) for qk, params in fp_pairs}
            output_tile_cat_list.append(TileCatalog(d))
        return output_tile_cat_list

    @torch.no_grad()
    def sample(self, x_cat, use_mode=False, filter_by_n_sources=True):
        tile_cat_list = self._individual_sample(x_cat, use_mode=use_mode)
        stacked_tile_cat = self._stack_tile_cat(tile_cat_list)
        if filter_by_n_sources:
            d = {}
            for k, v in stacked_tile_cat.items():
                if k == "n_sources":
                    d[k] = v
                else:
                    d[k] = v * stacked_tile_cat["n_sources_mask"]
            return TileCatalog(d)
        return stacked_tile_cat

    def compute_nll(self, x_cat, true_tile_cat, topk):
        x_tile_cat = self.sample(x_cat, use_mode=True, filter_by_n_sources=False)
        matched_true_tile_cat_list = self._match_tile_cat(
            x_tile_cat,
            true_tile_cat,
            topk,
        )
        fp_pairs_list = self._factor_param_pairs(x_cat)
        total_nll = 0
        for fp_pairs, matched_true_tile_cat in zip(fp_pairs_list, matched_true_tile_cat_list):
            cur_nll = []
            for qk, params in fp_pairs:
                nll = qk.compute_nll(params, matched_true_tile_cat)
                if qk.name == "n_sources":
                    n_sources = matched_true_tile_cat["n_sources"].bool()
                    nll = torch.where(n_sources, nll, 0.1 * nll)
                cur_nll.append(nll)
            total_nll += sum(cur_nll)
        return total_nll
