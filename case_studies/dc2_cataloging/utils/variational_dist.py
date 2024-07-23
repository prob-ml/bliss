import copy
from typing import List

import numpy as np
import torch
from einops import rearrange
from scipy.optimize import linear_sum_assignment

from bliss.catalog import TileCatalog
from bliss.encoder.variational_dist import BernoulliFactor, VariationalDist, VariationalFactor
from bliss.surveys.dc2 import unpack_dict


class BernoulliFactorReturnProbs(BernoulliFactor):
    def sample(self, params, use_mode=False):
        qk_probs = self._get_dist(params).probs
        qk_probs = rearrange(qk_probs, "b ht wt d -> b ht wt 1 d")
        return super().sample(params, use_mode), qk_probs


class VariationalDistReturnProbs(VariationalDist):
    def sample(self, x_cat, use_mode=False):
        fp_pairs = self._factor_param_pairs(x_cat)
        d = {}
        for qk, params in fp_pairs:
            if qk.name != "source_type":
                d[qk.name] = qk.sample(params, use_mode)
            else:
                assert isinstance(qk, BernoulliFactorReturnProbs), "wrong source_type class"
                d["source_type"], d["source_type_probs"] = qk.sample(params, use_mode)

        return TileCatalog(d)


class MultiVariationalDist(torch.nn.Module):
    def __init__(self, tile_slen: int, factors: List[VariationalFactor], repeat_times: int):
        super().__init__()

        self.tile_slen = tile_slen
        self.factors = factors
        self.repeat_times = repeat_times
        self.split_sizes = [v.n_params for v in self.factors]
        assert repeat_times > 1

    def _separate_x_cat(self, x_cat):
        chunk_size = sum(self.split_sizes)
        return torch.split(x_cat, chunk_size, 3)

    @classmethod
    def _union_tile_cat(cls, tile_cat_list: List[TileCatalog]):
        output_tile_cat = None
        for tile_cat in tile_cat_list:
            if output_tile_cat is None:
                output_tile_cat = tile_cat
            else:
                output_tile_cat = output_tile_cat.union(tile_cat, disjoint=False)

        return output_tile_cat

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
        return est_to_true_indices * one_to_one_match + offset

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
        return torch.from_numpy(est_to_true_indices).to(device=ori_device)

    @classmethod
    def _pad_max_sources(cls, v):
        return torch.cat((v, torch.zeros_like(v[:, :, :, 0, :].unsqueeze(-2))), dim=-2)

    def _match_tile_cat(
        self,
        true_tile_cat: TileCatalog,
        est_tile_cat: TileCatalog,
        use_linear_sum_assignment: bool = True,
    ):
        assert est_tile_cat.max_sources == true_tile_cat.max_sources
        assert est_tile_cat.tile_slen == true_tile_cat.tile_slen
        est_tile_dict = copy.copy(est_tile_cat.data)
        true_tile_dict = copy.copy(true_tile_cat.data)
        b, nth, ntw, m = est_tile_dict["locs"].shape[:-1]
        est_is_on_mask = rearrange(est_tile_cat.is_on_mask, "b nth ntw m -> b nth ntw m 1")
        true_is_on_mask = rearrange(true_tile_cat.is_on_mask, "b nth ntw m -> b nth ntw m 1")

        est_locs = est_tile_dict["locs"]
        true_locs = true_tile_dict["locs"]
        est_locs = torch.where(est_is_on_mask, est_locs, torch.full_like(est_locs, 1e2))
        true_locs = torch.where(true_is_on_mask, true_locs, torch.full_like(true_locs, 1e2))
        est_locs = rearrange(est_locs, "b nth ntw m k -> (b nth ntw) m k")
        true_locs = rearrange(true_locs, "b nth ntw m k -> (b nth ntw) m k")

        locs_distance = torch.cdist(est_locs, true_locs, p=2)  # (B, m, m)

        if not use_linear_sum_assignment:
            est_to_true_indices = self._fast_one_to_one_match(
                locs_distance, est_is_on_mask, true_is_on_mask
            )
        else:
            est_to_true_indices = self._linear_sum_assignment(locs_distance)

        est_to_true_indices = rearrange(
            est_to_true_indices, "(b nth ntw) m k -> b nth ntw m k", b=b, nth=nth, ntw=ntw
        )

        # build an padded tile dict due to the requirement of `_fast_one_to_one_match`
        # it's also compatible with `_linear_sum_assignment`
        padded_true_tile_dict = {
            k: self._pad_max_sources(v)
            for k, v in true_tile_dict.items()
            if k != "n_sources" and k in est_tile_dict
        }
        padded_true_tile_dict["n_sources"] = self._pad_max_sources(true_is_on_mask).to(
            dtype=true_tile_dict["n_sources"].dtype
        )
        target_tile_dict = {k: torch.zeros_like(v) for k, v in padded_true_tile_dict.items()}

        flat_index_offset = torch.arange(
            b * nth * ntw,
            dtype=est_to_true_indices.dtype,
            device=est_to_true_indices.device,
        )
        flat_index_offset = flat_index_offset.view(b, nth, ntw, 1, 1)
        flat_index_offset *= m + 1
        flat_est_to_true_indices = (est_to_true_indices + flat_index_offset).flatten(
            end_dim=-2
        )  # (:, 2)
        flat_est_indices = flat_est_to_true_indices[:, 0].tolist()
        flat_true_indices = flat_est_to_true_indices[:, 1].tolist()
        for k, v in target_tile_dict.items():
            ori_v_shape = v.shape
            v = v.view(-1, v.shape[-1])
            true_v = padded_true_tile_dict[k]
            true_v = true_v.view(-1, true_v.shape[-1])
            v[flat_est_indices, :] = true_v[flat_true_indices, :]
            v = v.view(*ori_v_shape)[:, :, :, :-1, :].unbind(dim=-2)
            target_tile_dict[k] = [sub_v.unsqueeze(-2) for sub_v in v]

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

    def sample(self, x_cat, use_mode=False):
        tile_cat_list = self._individual_sample(x_cat.detach(), use_mode=use_mode)
        return self._union_tile_cat(tile_cat_list)

    def compute_nll(self, x_cat, true_tile_cat):
        x_tile_cat = self.sample(x_cat, use_mode=True)
        matched_true_tile_cat_list = self._match_tile_cat(
            true_tile_cat, x_tile_cat, use_linear_sum_assignment=True
        )
        fp_pairs_list = self._factor_param_pairs(x_cat)
        total_nll = None
        for fp_pairs, matched_true_tile_cat in zip(fp_pairs_list, matched_true_tile_cat_list):
            cur_nll = sum(qk.compute_nll(params, matched_true_tile_cat) for qk, params in fp_pairs)
            if total_nll is not None:
                total_nll += cur_nll
            else:
                total_nll = cur_nll

        return total_nll
