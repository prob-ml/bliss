import torch
import torch.distributions as dist

from einops import rearrange

from bliss.catalog import TileCatalog
from bliss.encoder.variational_dist import VariationalDist as BlissVD
from bliss.encoder.variational_dist import SourcesGating, VariationalFactor


class MyBernoulli(dist.Bernoulli):
    def log_prob(self, value):
        return super().log_prob(value.float())


class BernoulliFactor(VariationalFactor):
    def __init__(self, *args, **kwargs):
        super().__init__(1, *args, **kwargs)
        self.dim = 1

    def get_dist(self, params):
        assert params.ndim == 4
        assert params.shape[-1] == 1  # (b, h, w, 1)
        yes_prob = params.sigmoid().clamp(1e-4, 1 - 1e-4).squeeze(-1)
        assert not yes_prob.isnan().any() and not yes_prob.isinf().any()
        # the bliss's BernoulliFactor uses Categorical whose mean returns nan
        return MyBernoulli(yes_prob)
    

class NormalLogFluxFactor(VariationalFactor):
    def __init__(self, *args, dim=6, **kwargs):
        self.dim = dim  # the dimension of a multivariate normal
        n_params = 2 * dim  # mean and std for each dimension (diagonal covariance)
        super().__init__(n_params, *args, **kwargs)

    def get_dist(self, params):
        mu = params[:, :, :, 0 : self.dim].clamp(-10, 30)
        sigma = params[:, :, :, self.dim : self.n_params].clamp(-20, 15).exp().sqrt()
        assert not mu.isnan().any() and not mu.isinf().any(), "mu contains invalid values"
        assert not sigma.isnan().any() and not sigma.isinf().any(), "sigma contains invalid values"
        iid_dist = dist.Normal(loc=mu, scale=sigma)
        return dist.Independent(iid_dist, 1)


class VariationalDist(BlissVD):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.factor_other_chs = [f.dim for f in self.factors[1:]]
        assert self.factors[0].name == "n_sources"
    
    @property
    def n_params_per_source(self):
        raise NotImplementedError()

    def sample(self, x_cat, use_mode=False, return_base_cat=False):
        raise NotImplementedError()
    
    @property
    def n_ns_params_per_source(self):
        return 1
    
    @property
    def n_other_params_per_source(self):
        return sum(f.n_params for f in self.factors[1:])
    
    @property
    def n_ns_chs_per_source(self):
        return 1

    @property
    def n_other_chs_per_source(self):
        return sum(self.factor_other_chs)

    def n_sources_mean(self, x_cat_n_sources: torch.Tensor):
        assert self.factors[0].name == "n_sources"
        assert isinstance(self.factors[0], BernoulliFactor)
        return self.factors[0].get_dist(x_cat_n_sources).mean

    # only use this func for inference
    # we assume the nll gating for other params is SoucesGating
    # we assume the empty position should have 0.0 at the end of inference
    def other_mean(self, x_cat_other: torch.Tensor, est_n_sources: torch.Tensor):
        assert est_n_sources.dtype == torch.bool
        assert x_cat_other.shape[:-1] == est_n_sources.shape  # (b, h, w, params), (b, h, w)
        assert self.factors[0].name == "n_sources"
        split_sizes = [v.n_params for v in self.factors[1:]]
        dist_params_list = torch.split(x_cat_other, split_sizes, 3)
        mean_list = []
        for f, p in zip(self.factors[1:], dist_params_list, strict=True):
            assert isinstance(f.nll_gating, SourcesGating)
            if f.name == "fluxes":
                assert isinstance(f, NormalLogFluxFactor)
            mean_list.append(f.get_dist(p).mean)
        mean_tensor = torch.cat(mean_list, dim=-1)  # (b, h, w, k)
        return torch.where(est_n_sources.unsqueeze(-1), mean_tensor, 0.0)
    
    def tile_cat_to_tensor(self, tile_catalog: TileCatalog):
        assert self.factors[0].name == "n_sources"
        n_sources = tile_catalog["n_sources"]  # (b, h, w)
        assert n_sources.max() <= 1
        other = []
        for f in self.factors[1:]:
            d = tile_catalog[f.name].squeeze(-2)
            if f.name != "fluxes":
                other.append(d)
            else:
                other.append(torch.log(d.clip(min=1.0)))
        other = torch.cat(other, dim=-1)  # (b, h, w, k)
        other = other * n_sources.unsqueeze(-1)
        return rearrange(n_sources.float(), "b h w -> b h w 1"), other
    
    def tensor_to_tile_cat(self, n_sources: torch.Tensor, other: torch.Tensor):
        assert self.factors[0].name == "n_sources"
        assert n_sources.dtype == torch.long
        assert n_sources.ndim == 3
        assert other.dtype == torch.float
        assert n_sources.max() <= 1
        assert n_sources.shape == other.shape[:-1]
        d = {"n_sources": n_sources}
        other_vars = torch.split(other, self.factor_other_chs, dim=-1)
        for f, other_v in zip(self.factors[1:], other_vars, strict=True):
            if f.name == "fluxes":
                d[f.name] = rearrange(torch.exp(other_v), "b h w k -> b h w 1 k")
            elif f.name == "locs":
                d[f.name] = rearrange(other_v.clip(0.0, 1.0), "b h w k -> b h w 1 k")
            else:
                raise NotImplementedError()
        return TileCatalog(d)
