import torch
from einops import rearrange

from bliss.catalog import TileCatalog
from bliss.encoder.variational_dist import BernoulliFactor, NllGating, VariationalDist


class MyBernoulliFactor(BernoulliFactor):
    def sample(self, params, use_mode=False):
        qk_probs = self._get_dist(params).probs
        qk_probs = rearrange(qk_probs, "b ht wt d -> b ht wt 1 d")
        return super().sample(params, use_mode), qk_probs


class MyVariationalDist(VariationalDist):
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

    def sample_vsbc(self, x_cat: torch.Tensor, true_tile_cat: TileCatalog, use_mode=False):
        assert true_tile_cat.max_sources == 1
        sample_result = self.sample(x_cat, use_mode=use_mode)
        assert sample_result.max_sources == 1
        fp_pairs = self._factor_param_pairs(x_cat)
        # for now, only test the vsbc for ellipticity and flux
        for qk, params in fp_pairs:
            match qk.name:
                case "ellipticity":
                    first_dist = qk.get_first_dist(params)
                    second_dist = qk.get_second_dist(params)
                    first_vsbc = 1 - first_dist.cdf(
                        true_tile_cat["ellipticity"][..., 0, 0].nan_to_num(0)
                    )
                    second_vsbc = 1 - second_dist.cdf(
                        true_tile_cat["ellipticity"][..., 0, 1].nan_to_num(0)
                    )
                    first_vsbc = rearrange(first_vsbc, "b nth ntw -> b nth ntw 1 1")
                    second_vsbc = rearrange(second_vsbc, "b nth ntw -> b nth ntw 1 1")
                    cosmodc2_mask = true_tile_cat["cosmodc2_mask"]
                    first_vsbc = torch.where(cosmodc2_mask, first_vsbc, torch.nan)
                    second_vsbc = torch.where(cosmodc2_mask, second_vsbc, torch.nan)
                    sample_result["ellipticity_vsbc"] = torch.cat((first_vsbc, second_vsbc), dim=-1)
                case "star_fluxes":
                    flux_dist = [qk.get_dist_at_dim(params, d) for d in range(qk.dim)]
                    flux_vsbc = [
                        1 - dist.cdf(true_tile_cat["star_fluxes"][..., 0, d])
                        for d, dist in enumerate(flux_dist)
                    ]
                    flux_vsbc = [
                        rearrange(vsbc, "b nth ntw -> b nth ntw 1 1") for vsbc in flux_vsbc
                    ]
                    star_mask = true_tile_cat.star_bools
                    flux_vsbc = [torch.where(star_mask, vsbc, torch.nan) for vsbc in flux_vsbc]
                    sample_result["star_fluxes_vsbc"] = torch.cat(flux_vsbc, dim=-1)
                case "galaxy_fluxes":
                    flux_dist = [qk.get_dist_at_dim(params, d) for d in range(qk.dim)]
                    flux_vsbc = [
                        1 - dist.cdf(true_tile_cat["galaxy_fluxes"][..., 0, d])
                        for d, dist in enumerate(flux_dist)
                    ]
                    flux_vsbc = [
                        rearrange(vsbc, "b nth ntw -> b nth ntw 1 1") for vsbc in flux_vsbc
                    ]
                    galaxy_mask = true_tile_cat.galaxy_bools
                    flux_vsbc = [torch.where(galaxy_mask, vsbc, torch.nan) for vsbc in flux_vsbc]
                    sample_result["galaxy_fluxes_vsbc"] = torch.cat(flux_vsbc, dim=-1)
        return sample_result


class Cosmodc2Gating(NllGating):
    @classmethod
    def __call__(cls, true_tile_cat: TileCatalog):
        return rearrange(true_tile_cat["cosmodc2_mask"], "b ht wt 1 1 -> b ht wt")
