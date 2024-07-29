from einops import rearrange

from bliss.catalog import TileCatalog
from bliss.encoder.variational_dist import BernoulliFactor, NllGating, VariationalDist


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


class Cosmodc2Gating(NllGating):
    @classmethod
    def __call__(cls, true_tile_cat: TileCatalog):
        return rearrange(true_tile_cat["cosmodc2_mask"], "b ht wt 1 1 -> b ht wt")
