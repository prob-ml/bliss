from abc import ABC, abstractmethod
from typing import List

import torch
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
        gating = self._loss_gating(true_tile_cat)
        if gating.shape != target.shape:
            assert gating.shape == target.shape[:-1]
            target = torch.where(gating.unsqueeze(-1), target, 0)
        else:
            target = torch.where(gating, target, 0)
        assert not torch.isnan(target).any()
        assert ((target >= -1.0) & (target <= 1.0)).all()
        return target

    def decode_params(self, params: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()

    def decode(self, params: torch.Tensor):
        assert ((params >= -1.0) & (params <= 1.0)).all()
        sample_cat_tensor = self.decode_params(params)
        if self.sample_rearrange is not None:
            sample_cat_tensor = rearrange(sample_cat_tensor, self.sample_rearrange)
        assert (
            sample_cat_tensor.isfinite().all()
        ), f"sample_cat has invalid values: {sample_cat_tensor}"
        return sample_cat_tensor

    def gating_loss(self, true_tile_cat: TileCatalog):
        return repeat(self._loss_gating(true_tile_cat), "b h w -> b h w k", k=self.n_params)


class CatalogParser(torch.nn.Module):
    def __init__(self, factors: List[DiffusionFactor]):
        super().__init__()
        self.factors = factors

    @property
    def n_params_per_source(self):
        return sum(fs.n_params for fs in self.factors)

    def encode(self, true_tile_cat: TileCatalog):
        output_tensor = []
        for factor in self.factors:
            name = factor.name
            encoded_v = factor.encode(true_tile_cat)
            if name == "n_sources":
                output_tensor.append(rearrange(encoded_v, "b h w -> b h w 1"))
            else:
                output_tensor.append(encoded_v)
        return torch.cat(output_tensor, dim=-1)

    def _factor_param_pairs(self, x_cat: torch.Tensor):
        split_sizes = [v.n_params for v in self.factors]
        dist_params_lst = torch.split(x_cat, split_sizes, 3)
        return zip(self.factors, dist_params_lst)

    def decode(self, x_cat: torch.Tensor):
        fp_pairs = self._factor_param_pairs(x_cat)
        d = {qk.name: qk.decode(params) for qk, params in fp_pairs}
        return TileCatalog(d)

    def gating_loss(self, loss: torch.Tensor, true_tile_cat: TileCatalog):
        loss_gating = torch.cat(
            [f.gating_loss(true_tile_cat) for f in self.factors], dim=-1
        )  # (b, h, w, k)
        assert loss.shape == loss_gating.shape
        return loss * loss_gating

    def get_loss_gating(self, true_tile_cat):
        return torch.cat([f.gating_loss(true_tile_cat) for f in self.factors], dim=-1)

    def factor_param_tensor(self, input_tensor):
        split_sizes = [v.n_params for v in self.factors]
        return torch.split(input_tensor, split_sizes, 3)


class NormalizedFactor(DiffusionFactor):
    def __init__(self, *args, data_min, data_max, **kwargs):
        super().__init__(*args, **kwargs)
        self.data_min = data_min
        self.data_max = data_max

    def encode_tile_cat(self, true_tile_cat):
        assert true_tile_cat[self.name].shape[-2] == 1
        assert true_tile_cat[self.name].shape[-1] == self.n_params
        target = true_tile_cat[self.name][..., 0, :]  # (b, h, w, k)
        return (target - self.data_min) / (self.data_max - self.data_min) * 2 - 1

    def decode_params(self, params):
        return (params + 1) / 2 * (self.data_max - self.data_min) + self.data_min


class LogNormalizedFactor(DiffusionFactor):
    def __init__(self, *args, data_min, log_data_min, log_data_max, **kwargs):
        super().__init__(*args, **kwargs)
        self.data_min = data_min
        self.log_data_min = log_data_min
        self.log_data_max = log_data_max

    def encode_tile_cat(self, true_tile_cat):
        assert true_tile_cat[self.name].shape[-2] == 1
        assert true_tile_cat[self.name].shape[-1] == self.n_params
        target = true_tile_cat[self.name][..., 0, :]  # (b, h, w, k)
        log_target = torch.log(target - self.data_min + 1)
        return (log_target - self.log_data_min) / (self.log_data_max - self.log_data_min) * 2 - 1

    def decode_params(self, params):
        log_params = (params + 1) / 2 * (self.log_data_max - self.log_data_min) + self.log_data_min
        return torch.exp(log_params) - 1 + self.data_min


class OneBitFactor(DiffusionFactor):
    def __init__(self, *args, bit_value, **kwargs):
        super().__init__(*args, **kwargs)
        self.bit_value = bit_value

    def encode_tile_cat(self, true_tile_cat):
        if self.name == "n_sources":
            target = true_tile_cat[self.name]  # (b, h, w)
        else:
            assert true_tile_cat[self.name].shape[-2] == 1
            assert true_tile_cat[self.name].shape[-1] == self.n_params
            target = true_tile_cat[self.name][..., 0, :]  # (b, h, w, k)
        return torch.where(target > 0, self.bit_value, -self.bit_value)

    def decode_params(self, params):
        return (params > 0).int()
