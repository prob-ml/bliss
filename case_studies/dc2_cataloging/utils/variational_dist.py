import torch
from einops import rearrange
from torch.distributions import Independent

from bliss.catalog import TileCatalog
from bliss.encoder.variational_dist import (
    BernoulliFactor,
    NllGating,
    TruncatedDiagonalMVN,
    VariationalDist,
)


class Cosmodc2Gating(NllGating):
    @classmethod
    def __call__(cls, true_tile_cat: TileCatalog):
        return rearrange(true_tile_cat["cosmodc2_mask"], "b ht wt 1 1 -> b ht wt")


class MyBernoulliFactor(BernoulliFactor):
    def sample(self, params, use_mode=False):
        qk_probs = self.get_dist(params).probs
        qk_probs = rearrange(qk_probs, "b ht wt d -> b ht wt 1 d")
        return super().sample(params, use_mode), qk_probs


class MyBasicVariationalDist(VariationalDist):
    def sample(self, x_cat, use_mode=False):
        fp_pairs = self._factor_param_pairs(x_cat)
        d = {}
        for qk, params in fp_pairs:
            if qk.name == "source_type":
                assert isinstance(qk, MyBernoulliFactor), "wrong source_type class"
                d["source_type"], d["source_type_probs"] = qk.sample(params, use_mode)
            elif qk.name == "n_sources":
                assert isinstance(qk, MyBernoulliFactor), "wrong b_sources class"
                d["n_sources"], d["n_sources_probs"] = qk.sample(params, use_mode)
            else:
                d[qk.name] = qk.sample(params, use_mode)
        return TileCatalog(d)


class CalibrationVariationalDist(MyBasicVariationalDist):
    def __init__(self, factors, tile_slen, ci_lower_q, ci_upper_q):
        super().__init__(factors, tile_slen)
        assert ci_lower_q < ci_upper_q
        self.ci_lower_q = ci_lower_q
        self.ci_upper_q = ci_upper_q

    @classmethod
    def _get_vsbc(cls, qk, params, true_tile_cat):
        dist = qk.get_dist(params)
        if isinstance(dist, Independent):
            dist = dist.base_dist

        true_value = true_tile_cat[qk.name][..., 0, :]
        if qk.name == "ellipticity":
            true_value = true_value.nan_to_num(0)

        if isinstance(dist, TruncatedDiagonalMVN):
            vsbc = 1 - dist.event_cdf(true_value)
        else:
            vsbc = 1 - dist.cdf(true_value)
        vsbc = rearrange(vsbc, "b nth ntw d -> b nth ntw 1 d")

        nll_gating = qk.nll_gating(true_tile_cat)
        nll_gating = rearrange(nll_gating, "b nth ntw -> b nth ntw 1 1")
        return torch.where(nll_gating, vsbc, torch.nan)

    @classmethod
    def _get_credible_interval(cls, qk, params, lower_q, upper_q):
        dist = qk.get_dist(params)
        if isinstance(dist, Independent):
            dist = dist.base_dist

        dist_shape = dist.batch_shape + dist.event_shape
        lower_q_tensor = torch.full(dist_shape, fill_value=lower_q, device=params.device)
        upper_q_tensor = torch.full(dist_shape, fill_value=upper_q, device=params.device)
        if isinstance(dist, TruncatedDiagonalMVN):
            lower_b = dist.event_icdf(lower_q_tensor)
            upper_b = dist.event_icdf(upper_q_tensor)
        else:
            lower_b = dist.icdf(lower_q_tensor)
            upper_b = dist.icdf(upper_q_tensor)
        lower_b = rearrange(lower_b, "b nth ntw d -> b nth ntw 1 d")
        upper_b = rearrange(upper_b, "b nth ntw d -> b nth ntw 1 d")
        return lower_b, upper_b

    def sample(
        self,
        x_cat: torch.Tensor,
        use_mode=False,
        return_base_cat=False,
        true_tile_cat: TileCatalog = None,
    ):
        sample_result = super().sample(x_cat, use_mode=use_mode)

        if true_tile_cat is not None:
            assert true_tile_cat.max_sources == 1
            assert sample_result.max_sources == 1
            fp_pairs = self._factor_param_pairs(x_cat)
            # for now, only test the vsbc for locs, ellipticity and flux
            vsbc_variables = {
                "locs",
                "ellipticity",
                "fluxes",
            }
            credible_interval_variables = {
                "locs",
                "ellipticity",
                "fluxes",
            }
            for qk, params in fp_pairs:
                if qk.name in vsbc_variables:
                    sample_result[qk.name + "_vsbc"] = self._get_vsbc(qk, params, true_tile_cat)
                if qk.name in credible_interval_variables:
                    lower_b, upper_b = self._get_credible_interval(
                        qk, params, self.ci_lower_q, self.ci_upper_q
                    )
                    sample_result[qk.name + "_ci_lower"] = lower_b
                    sample_result[qk.name + "_ci_upper"] = upper_b

        return sample_result
