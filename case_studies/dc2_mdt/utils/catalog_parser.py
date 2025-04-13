from abc import ABC, abstractmethod
from typing import List

import torch
import math
import logging
from einops import rearrange, repeat

from bliss.catalog import TileCatalog


class LossGating(ABC):
    @classmethod
    @abstractmethod
    def __call__(cls, true_tile_cat: TileCatalog):
        """Get Gating for Loss."""


class NullGating(LossGating):
    @classmethod
    def __call__(cls, true_tile_cat: TileCatalog):
        tc_keys = true_tile_cat.keys()
        if "n_sources" in tc_keys:
            return torch.ones_like(true_tile_cat["n_sources"]).bool()
        first = true_tile_cat[list(tc_keys)[0]]
        return torch.ones(first.shape[:-1]).bool().to(first.device)
    
class WeightedNSourcesGating(LossGating):
    @classmethod
    def __call__(cls, true_tile_cat: TileCatalog):
        all_ones_mask = torch.ones_like(true_tile_cat["n_sources"])
        emphasis_mask = (true_tile_cat["n_sources"] > 0) * 5
        return all_ones_mask + emphasis_mask

class SourcesGating(LossGating):
    @classmethod
    def __call__(cls, true_tile_cat: TileCatalog):
        return true_tile_cat["n_sources"].bool()


class Cosmodc2Gating(LossGating):
    @classmethod
    def __call__(cls, true_tile_cat: TileCatalog):
        return rearrange(true_tile_cat["cosmodc2_mask"], "b ht wt 1 1 -> b ht wt")


class DiffusionFactor:
    def __init__(
        self,
        n_params: int,
        name: str,
        sample_rearrange: str = None,
        loss_gating: LossGating = None,
    ):
        self.name = name
        self.n_params = n_params
        self.sample_rearrange = sample_rearrange
        if loss_gating is None:
            self._loss_gating = NullGating()
        elif issubclass(type(loss_gating), LossGating):
            self._loss_gating = loss_gating
        else:
            raise TypeError("invalid loss_gating type")

    def encode_tile_cat(self, true_tile_cat: TileCatalog) -> torch.Tensor:
        raise NotImplementedError()

    def encode(self, true_tile_cat: TileCatalog):
        target = self.encode_tile_cat(true_tile_cat)
        assert torch.isfinite(target).all()
        assert not torch.isnan(target).any()
        return target

    def decode_params(self, params: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()

    def decode(self, params: torch.Tensor):
        sample_cat_tensor = self.decode_params(params)
        if self.sample_rearrange is not None:
            sample_cat_tensor = rearrange(sample_cat_tensor, self.sample_rearrange)
        assert (
            sample_cat_tensor.isfinite().all()
        ), f"sample_cat has invalid values: {sample_cat_tensor}"
        assert not torch.isnan(sample_cat_tensor).any()
        return sample_cat_tensor

    def get_gating_for_loss(self, true_tile_cat: TileCatalog):
        return repeat(self._loss_gating(true_tile_cat), "b h w -> b h w k", k=self.n_params)

    def clip_tensor(self, input_tensor: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()

    def craft_fake_data(self, on_mask: torch.Tensor, dtype=None):
        raise NotImplementedError()
    
    def invalid_points_warning(self):
        logger = logging.getLogger("DiffusionFactor")
        warning_msg = f"WARNING: find invalid values in input catalog for factor [{self.name}]"
        logger.warning(warning_msg)


class CatalogParser(torch.nn.Module):
    def __init__(self, factors: List[DiffusionFactor]):
        super().__init__()
        self.factors = factors
        # ensure the first factor is n_sources
        # this is designed for the "empty_tile_random_noise" in diffusion model
        # "craft_fake_data" also needs this feature
        if self.factors[0].name != "n_sources" and self.factors[0].name != "n_sources_multi":
            logger = logging.getLogger("CatalogParser")
            warning_msg = "WARNING: the first factor is neither 'n_sources' nor 'n_sources_multi'; please make sure this is intended"
            logger.warning(warning_msg)

    @property
    def n_params_per_source(self):
        return sum(fs.n_params for fs in self.factors)

    def encode(self, true_tile_cat: TileCatalog):
        output_tensor = []
        for factor in self.factors:
            encoded_v = factor.encode(true_tile_cat)
            output_tensor.append(encoded_v)
        return torch.cat(output_tensor, dim=-1)

    def _factor_param_pairs(self, x_cat: torch.Tensor):
        split_sizes = [v.n_params for v in self.factors]
        dist_params_lst = torch.split(x_cat, split_sizes, 3)
        return zip(self.factors, dist_params_lst)

    def decode(self, x_cat: torch.Tensor):
        assert len(x_cat.shape) == 4  # (b, h, w, k)
        fp_pairs = self._factor_param_pairs(x_cat)
        d = {}
        for qk, params in fp_pairs:
            d[qk.name] = qk.decode(params)
            if isinstance(qk, OneBitNSourcesLocsFactor):
                d["locs"] = repeat(d[qk.name] * 0.5, "b h w -> b h w 1 k", k=2)
            if isinstance(qk, MultiBitsNSourcesFactor):
                d["n_sources"] = rearrange((d[qk.name] > 0).to(dtype=d[qk.name].dtype),
                                           "b h w 1 1 -> b h w")
                d["n_sources_multi"] = d[qk.name]  # (b, h, w, 1, 1)
        if "locs" in d:
            return TileCatalog(d)
        else:
            # for cases where we don't predict locs
            return d

    def gating_loss(self, loss: torch.Tensor, true_tile_cat: TileCatalog):
        loss_gating = torch.cat(
            [f.get_gating_for_loss(true_tile_cat) for f in self.factors], dim=-1
        )  # (b, h, w, k)
        assert loss.shape == loss_gating.shape
        return loss * loss_gating

    def get_gating_for_loss(self, true_tile_cat):
        return torch.cat([f.get_gating_for_loss(true_tile_cat) for f in self.factors], dim=-1)

    def factor_tensor(self, input_tensor):
        assert len(input_tensor.shape) == 4
        split_sizes = [v.n_params for v in self.factors]
        assert input_tensor.shape[3] == sum(split_sizes)
        return torch.split(input_tensor, split_sizes, 3)
    
    def clip_tensor(self, input_tensor):
        assert len(input_tensor.shape) == 4
        factored_input_tensor = self.factor_tensor(input_tensor)
        return torch.cat([
            f.clip_tensor(ft) for f, ft in zip(self.factors, factored_input_tensor)
        ], dim=3)


class NormalizedFactor(DiffusionFactor):
    def __init__(self, *args, data_min, data_max, scale, latent_zero_point, **kwargs):
        super().__init__(*args, **kwargs)
        self.data_min = data_min
        self.data_max = data_max
        self.scale = scale
        self.latent_zero_point = latent_zero_point
        self.invalid_point_flag = False

    def encode_tile_cat(self, true_tile_cat):
        assert len(true_tile_cat[self.name].shape) == 5  # (b, h, w, 1, k)
        assert true_tile_cat[self.name].shape[-2] == 1
        assert true_tile_cat[self.name].shape[-1] == self.n_params
        target_n_sources = true_tile_cat["n_sources"]  # (b, h, w)
        target = true_tile_cat[self.name][..., 0, :]  # (b, h, w, k)
        invalid_points = torch.isnan(target) | torch.isinf(target)
        if (not self.invalid_point_flag) and invalid_points.any():
            self.invalid_point_flag = True
            self.invalid_points_warning()
        target_minus_1_to_1 = (target - self.data_min) / (self.data_max - self.data_min) * 2 - 1
        masked_target_minus_1_to_1 = torch.where(invalid_points | (target_n_sources == 0).unsqueeze(-1), 
                                                 torch.ones_like(target_minus_1_to_1) * self.latent_zero_point, 
                                                 target_minus_1_to_1)
        assert ((masked_target_minus_1_to_1 >= -1.0) & (masked_target_minus_1_to_1 <= 1.0)).all()
        return target_minus_1_to_1 * self.scale

    def decode_params(self, params):
        assert ((params >= -self.scale) & (params <= self.scale)).all()
        params_minus_1_to_1 = params / self.scale
        return (params_minus_1_to_1 + 1) / 2 * (self.data_max - self.data_min) + self.data_min
    
    def clip_tensor(self, input_tensor):
        return torch.clamp(input_tensor, min=-self.scale, max=self.scale)


class LogNormalizedFactor(DiffusionFactor):
    def __init__(self, *args, data_min, log_data_min, log_data_max, scale, latent_zero_point, **kwargs):
        super().__init__(*args, **kwargs)
        self.data_min = data_min
        self.log_data_min = log_data_min
        self.log_data_max = log_data_max
        self.scale = scale
        self.latent_zero_point = latent_zero_point
        self.invalid_point_flag = False

    def encode_tile_cat(self, true_tile_cat):
        assert len(true_tile_cat[self.name].shape) == 5  # (b, h, w, 1, k)
        assert true_tile_cat[self.name].shape[-2] == 1
        assert true_tile_cat[self.name].shape[-1] == self.n_params
        target = true_tile_cat[self.name][..., 0, :]  # (b, h, w, k)
        target_n_sources = true_tile_cat["n_sources"]  # (b, h, w)
        invalid_points = torch.isnan(target) | torch.isinf(target)
        if (not self.invalid_point_flag) and invalid_points.any():
            self.invalid_point_flag = True
            self.invalid_points_warning()
        log_target = torch.log1p(target - self.data_min)
        log_target_minus_1_to_1 = (log_target - self.log_data_min) / (self.log_data_max - self.log_data_min) * 2 - 1
        masked_log_target_minus_1_to_1 = torch.where(invalid_points | (target_n_sources == 0).unsqueeze(-1),
                                                     torch.ones_like(log_target_minus_1_to_1) * self.latent_zero_point,
                                                     log_target_minus_1_to_1)
        assert ((masked_log_target_minus_1_to_1 >= -1.0) & (masked_log_target_minus_1_to_1 <= 1.0)).all()
        return log_target_minus_1_to_1 * self.scale

    def decode_params(self, params):
        assert ((params >= -self.scale) & (params <= self.scale)).all()
        log_params_minus_1_to_1 = params / self.scale
        log_params = (log_params_minus_1_to_1 + 1) / 2 * (self.log_data_max - self.log_data_min) + self.log_data_min
        return torch.expm1(log_params) + self.data_min
    
    def clip_tensor(self, input_tensor):
        return torch.clamp(input_tensor, min=-self.scale, max=self.scale)


class OneBitFactor(DiffusionFactor):
    def __init__(self, *args, bit_value, threshold, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.n_params == 1
        self.bit_value = bit_value
        self.threshold = threshold
        assert self.threshold > -self.bit_value and self.threshold < self.bit_value

    def encode_tile_cat(self, true_tile_cat):
        if self.name == "n_sources":
            assert len(true_tile_cat[self.name].shape) == 3
            target = true_tile_cat[self.name]  # (b, h, w)
            return torch.where(target > 0, self.bit_value, -self.bit_value).unsqueeze(-1)
        assert len(true_tile_cat[self.name].shape) == 5  # (b, h, w, 1, k)
        assert true_tile_cat[self.name].shape[-2] == 1
        assert true_tile_cat[self.name].shape[-1] == self.n_params
        target = true_tile_cat[self.name][..., 0, :]  # (b, h, w, k)
        return torch.where(target > 0, self.bit_value, -self.bit_value)

    def decode_params(self, params):
        assert ((params >= -self.bit_value) & (params <= self.bit_value)).all()
        return (params > self.threshold).int()

    def clip_tensor(self, input_tensor):
        return torch.clamp(input_tensor, min=-self.bit_value, max=self.bit_value)
    
    
class OneBitNSourcesLocsFactor(OneBitFactor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.name == "n_sources"
    
def ints_to_bits(ints_tensor: torch.Tensor, num_bits: int):
    assert len(ints_tensor.shape) == 4  # (b, h, w, 1)
    assert ints_tensor.shape[-1] == 1 
    assert not torch.is_floating_point(ints_tensor)
    assert ints_tensor.min() >= 0
    assert ints_tensor.max() < 2 ** num_bits
    b, h, w, _  = ints_tensor.shape
    flat_ints = rearrange(ints_tensor, "b h w 1 -> (b h w) 1")
    flat_bits = (flat_ints >> torch.arange(num_bits - 1, -1, -1, 
                                           device=flat_ints.device)) & 1
    return rearrange(flat_bits, "(b h w) num_bits -> b h w num_bits", 
                     b=b, h=h, w=w, num_bits=num_bits)

def bits_to_ints(bits_tensor: torch.Tensor):
    assert len(bits_tensor.shape) == 4
    assert not torch.is_floating_point(bits_tensor)
    assert bits_tensor.min() >= 0
    assert bits_tensor.max() <= 1
    b, h, w, num_bits = bits_tensor.shape
    flat_bits = rearrange(bits_tensor, "b h w num_bits -> (b h w) num_bits")
    flat_ints = (flat_bits * 2 ** torch.arange(num_bits - 1, -1, -1, 
                                               device=flat_bits.device)).sum(dim=-1, 
                                                                             keepdim=True)
    return rearrange(flat_ints, "(b h w) 1 -> b h w 1",
                     b=b, h=h, w=w)


class MultiBitsNSourcesFactor(DiffusionFactor):
    def __init__(self, *args, bit_value, threshold, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.name == "n_sources_multi"
        self.bit_value = bit_value
        self.threshold = threshold
        assert self.threshold > -self.bit_value and self.threshold < self.bit_value

    def encode_tile_cat(self, true_tile_cat):
        target = true_tile_cat[self.name]  # (b, h, w, 1, 1)
        assert len(target.shape) == 5
        assert target.shape[-1] == 1
        assert target.shape[-2] == 1
        target = target.squeeze(-2)  # (b, h, w, 1)
        ns_bits = ints_to_bits(target, num_bits=self.n_params)  # (b, h, w, k)
        return torch.where(ns_bits > 0, self.bit_value, -self.bit_value)
    
    def decode_params(self, params):
        assert ((params >= -self.bit_value) & (params <= self.bit_value)).all()
        assert params.shape[-1] == self.n_params
        bits_tensor = (params > self.threshold).int()  # (b, h, w, k)
        return bits_to_ints(bits_tensor)  # (b, h, w, 1)
    
    def clip_tensor(self, input_tensor):
        return torch.clamp(input_tensor, min=-self.bit_value, max=self.bit_value)

    
class MultiBitsLocsFactor(DiffusionFactor):
    def __init__(self, *args, bit_value, one_side_parts, individual_hw, threshold, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.name == "locs"
        assert isinstance(one_side_parts, int)
        assert one_side_parts > 1
        self.one_side_parts = one_side_parts
        self.individual_hw = individual_hw
        self.one_part_len = 1.0 / self.one_side_parts
        assert self.n_params == int(math.log2(self.one_side_parts ** 2))
        if self.individual_hw:
            assert self.n_params % 2 == 0
        self.bit_value = bit_value
        self.threshold = threshold
        assert self.threshold > -self.bit_value and self.threshold < self.bit_value

    def encode_tile_cat(self, true_tile_cat):
        target = true_tile_cat[self.name]  # (b, h, w, 1, 2)
        assert len(target.shape) == 5
        assert target.shape[-1] == 2
        assert target.shape[-2] == 1
        target = target.squeeze(-2)  # (b, h, w, 2)
        assert (target <= 1.0).all()
        assert (target >= 0.0).all()
        end_point = torch.ones_like(target)
        cornor_points = target == end_point
        subtile_coords = torch.div(target, self.one_part_len, 
                                   rounding_mode="trunc").int()
        subtile_coords -= cornor_points.int()
        if not self.individual_hw:
            subtile_indices = subtile_coords[..., 0:1] * self.one_side_parts + \
                              subtile_coords[..., 1:2]  # (b, h, w, 1)
            assert subtile_indices.max() < self.one_side_parts ** 2
            assert subtile_indices.min() >= 0
            subtile_bits = ints_to_bits(subtile_indices, num_bits=self.n_params)  # (b, h, w, num_bits)
        else:
            assert subtile_coords.max() < self.one_side_parts
            assert subtile_coords.min() >= 0
            subtile_h_bits = ints_to_bits(subtile_coords[..., 0:1], num_bits=self.n_params // 2)
            subtile_w_bits = ints_to_bits(subtile_coords[..., 1:2], num_bits=self.n_params // 2)
            subtile_bits = torch.cat([subtile_h_bits, subtile_w_bits], dim=-1)
        return torch.where(subtile_bits > 0, self.bit_value, -self.bit_value)
    
    def decode_params(self, params):
        assert ((params >= -self.bit_value) & (params <= self.bit_value)).all()
        bits_tensor = (params > self.threshold).int()  # (b, h, w, num_bits)
        if not self.individual_hw:
            subtile_indices = bits_to_ints(bits_tensor)  # (b, h, w, 1)
            subtile_coords = torch.cat([
                subtile_indices // self.one_side_parts,
                subtile_indices % self.one_side_parts
            ], dim=-1)  # (b, h, w, 2)
        else:
            subtile_h_indices = bits_to_ints(bits_tensor[..., 0:(self.n_params // 2)])
            subtile_w_indices = bits_to_ints(bits_tensor[..., (self.n_params // 2):self.n_params])
            subtile_coords = torch.cat([subtile_h_indices, subtile_w_indices], dim=-1)
        return subtile_coords * self.one_part_len + self.one_part_len / 2

    def clip_tensor(self, input_tensor):
        return torch.clamp(input_tensor, min=-self.bit_value, max=self.bit_value)

