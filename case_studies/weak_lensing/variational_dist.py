"""Local variational distribution classes for weak lensing case study.

This module contains copies of the variational distribution classes from
bliss.encoder.variational_dist, modified to return plain dicts instead of
BaseTileCatalog/TileCatalog objects.
"""

from abc import ABC, abstractmethod

import torch
from torch.distributions import Independent, Normal


class VariationalDist(torch.nn.Module):
    def __init__(self, factors, tile_slen):
        super().__init__()
        self.factors = factors
        self.tile_slen = tile_slen

    @property
    def n_params_per_source(self):
        return sum(fs.n_params for fs in self.factors)

    def _factor_param_pairs(self, x_cat):
        split_sizes = [v.n_params for v in self.factors]
        dist_params_lst = torch.split(x_cat, split_sizes, 3)
        return zip(self.factors, dist_params_lst, strict=True)

    def sample(self, x_cat, use_mode=False):
        """Sample from the variational distribution.

        Returns a plain dict instead of BaseTileCatalog/TileCatalog.
        """
        fp_pairs = self._factor_param_pairs(x_cat)
        return {qk.name: qk.sample(params, use_mode) for qk, params in fp_pairs}

    def compute_nll(self, x_cat, true_cat):
        """Compute negative log-likelihood.

        Args:
            x_cat: Network output parameters
            true_cat: Plain dict with true values
        """
        fp_pairs = self._factor_param_pairs(x_cat)
        return sum(qk.compute_nll(params, true_cat) for qk, params in fp_pairs)


class NllGating(ABC):
    @classmethod
    @abstractmethod
    def __call__(cls, true_cat: dict):
        """Get Gating for nll."""


class NullGating(NllGating):
    @classmethod
    def __call__(cls, true_cat: dict):
        tc_keys = true_cat.keys()
        if "n_sources" in tc_keys:
            return torch.ones_like(true_cat["n_sources"]).bool()
        first = true_cat[next(iter(tc_keys))]
        return torch.ones(first.shape[:-1]).bool().to(first.device)


class VariationalFactor:
    def __init__(
        self,
        n_params,
        name,
        sample_rearrange=None,
        nll_rearrange=None,
        nll_gating=None,
    ):
        self.name = name
        self.n_params = n_params
        self.sample_rearrange = sample_rearrange
        self.nll_rearrange = nll_rearrange
        self.nll_gating = nll_gating if nll_gating is not None else NullGating()

    def sample(self, params, use_mode=False):
        qk = self.get_dist(params)
        sample_cat = qk.mode if use_mode else qk.sample()
        if self.sample_rearrange is not None:
            from einops import rearrange

            sample_cat = rearrange(sample_cat, self.sample_rearrange)
            assert sample_cat.isfinite().all(), f"sample_cat has invalid values: {sample_cat}"
        return sample_cat

    def compute_nll(self, params, true_cat):
        """Compute negative log-likelihood for this factor.

        Args:
            params: Network output parameters for this factor
            true_cat: Plain dict with true values
        """
        target = true_cat[self.name]
        if self.nll_rearrange is not None:
            from einops import rearrange

            target = rearrange(target, self.nll_rearrange)

        gating = self.nll_gating(true_cat)

        qk = self.get_dist(params)
        if gating.shape != target.shape:
            assert gating.shape == target.shape[:-1]
            target = torch.where(gating.unsqueeze(-1), target, 0)
        assert not torch.isnan(target).any()
        ungated_nll = -qk.log_prob(target)
        if ungated_nll.dim() == target.dim():  # (b, w, h, 1) -> (b,w,h) silent error
            ungated_nll = ungated_nll.squeeze(-1)
        return ungated_nll * gating


class NormalFactor(VariationalFactor):
    def __init__(self, *args, low_clamp=-20, high_clamp=20, **kwargs):
        super().__init__(2, *args, **kwargs)
        self.low_clamp = low_clamp
        self.high_clamp = high_clamp

    def get_dist(self, params):
        mean = params[:, :, :, 0:1]
        sd = params[:, :, :, 1:2].clamp(self.low_clamp, self.high_clamp).exp().sqrt()
        return Normal(mean, sd)


class IndependentMVNFactor(VariationalFactor):
    def __init__(self, *args, dim, low_clamp=-20, high_clamp=20, **kwargs):
        super().__init__(dim + dim, *args, **kwargs)
        self.dim = dim
        self.low_clamp = low_clamp
        self.high_clamp = high_clamp

    def get_dist(self, params):
        mean = params[:, :, :, : self.dim]
        sd = params[:, :, :, self.dim :].clamp(self.low_clamp, self.high_clamp).exp().sqrt()
        return Independent(Normal(mean, sd), 1)
