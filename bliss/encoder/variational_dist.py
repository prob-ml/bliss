import logging
from abc import ABC, abstractmethod

import torch
from einops import rearrange
from torch.distributions import (
    AffineTransform,
    Categorical,
    Distribution,
    Independent,
    LogNormal,
    Normal,
    SigmoidTransform,
    TransformedDistribution,
)

from bliss.catalog import TileCatalog


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
        return zip(self.factors, dist_params_lst)

    def sample(self, x_cat, use_mode=False):
        fp_pairs = self._factor_param_pairs(x_cat)
        d = {qk.name: qk.sample(params, use_mode) for qk, params in fp_pairs}
        return TileCatalog(d)

    def compute_nll(self, x_cat, true_tile_cat):
        fp_pairs = self._factor_param_pairs(x_cat)
        return sum(qk.compute_nll(params, true_tile_cat) for qk, params in fp_pairs)


class NllGating(ABC):
    @classmethod
    @abstractmethod
    def __call__(cls, true_tile_cat: TileCatalog):
        """Get Gating for nll."""


class NullGating(NllGating):
    @classmethod
    def __call__(cls, true_tile_cat: TileCatalog):
        return torch.ones_like(true_tile_cat["n_sources"]).bool()


class SourcesGating(NllGating):
    @classmethod
    def __call__(cls, true_tile_cat: TileCatalog):
        return true_tile_cat["n_sources"].bool()


class StarGating(NllGating):
    @classmethod
    def __call__(cls, true_tile_cat: TileCatalog):
        return rearrange(true_tile_cat.star_bools, "b ht wt 1 1 -> b ht wt")


class GalaxyGating(NllGating):
    @classmethod
    def __call__(cls, true_tile_cat: TileCatalog):
        return rearrange(true_tile_cat.galaxy_bools, "b ht wt 1 1 -> b ht wt")


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
        if nll_gating is None:
            self.nll_gating = NullGating()
        elif isinstance(nll_gating, str):
            self.nll_gating = self._get_nll_gating_instance(nll_gating)
        elif issubclass(type(nll_gating), NllGating):
            self.nll_gating = nll_gating
        else:
            raise TypeError("invalid nll_gating type")

    # to be compatible with the old yaml files
    def _get_nll_gating_instance(self, nll_gating: str):
        logger = logging.getLogger("VariationalFactor")
        warning_msg = "WARNING: please don't use str as nll_gating; it will be deprecated"
        logger.warning(warning_msg)
        if nll_gating == "n_sources":
            return SourcesGating()
        if nll_gating == "is_star":
            return StarGating()
        if nll_gating == "is_galaxy":
            return GalaxyGating()
        raise ValueError("invalide nll_gating string")

    def sample(self, params, use_mode=False):
        qk = self._get_dist(params)
        sample_cat = qk.mode if use_mode else qk.sample()
        if self.sample_rearrange is not None:
            sample_cat = rearrange(sample_cat, self.sample_rearrange)
            assert sample_cat.isfinite().all(), f"sample_cat has invalid values: {sample_cat}"
        return sample_cat

    def compute_nll(self, params, true_tile_cat):
        target = true_tile_cat[self.name]
        if self.nll_rearrange is not None:
            target = rearrange(target, self.nll_rearrange)

        gating = self.nll_gating(true_tile_cat)

        qk = self._get_dist(params)
        if gating.shape != target.shape:
            assert gating.shape == target.shape[:-1]
            target = torch.where(gating.unsqueeze(-1), target, 0)
        assert not torch.isnan(target).any()
        ungated_nll = -qk.log_prob(target)
        return ungated_nll * gating


class BernoulliFactor(VariationalFactor):
    def __init__(self, *args, **kwargs):
        super().__init__(1, *args, **kwargs)

    def _get_dist(self, params):
        yes_prob = params.sigmoid().clamp(1e-4, 1 - 1e-4)
        no_yes_prob = torch.cat([1 - yes_prob, yes_prob], dim=3)
        # this next line may be helpful with nans encountered during training with fp16s
        no_yes_prob = no_yes_prob.nan_to_num(nan=0.5)
        return Categorical(no_yes_prob)


class NormalFactor(VariationalFactor):
    def __init__(self, *args, low_clamp=-20, high_clamp=20, **kwargs):
        super().__init__(2, *args, **kwargs)
        self.low_clamp = low_clamp
        self.high_clamp = high_clamp

    def _get_dist(self, params):
        mean = params[:, :, :, 0]
        sd = params[:, :, :, 1].clamp(self.low_clamp, self.high_clamp).exp().sqrt()
        return Normal(mean, sd)


class BivariateNormalFactor(VariationalFactor):
    def __init__(self, *args, low_clamp=-20, high_clamp=20, **kwargs):
        super().__init__(4, *args, **kwargs)
        self.low_clamp = low_clamp
        self.high_clamp = high_clamp

    def _get_dist(self, params):
        mean = params[:, :, :, :2]
        sd = params[:, :, :, 2:].clamp(self.low_clamp, self.high_clamp).exp().sqrt()

        return Independent(Normal(mean, sd), 1)


class TDBNFactor(VariationalFactor):
    """Produces truncated bivariate normal distributions from unconstrained parameters."""

    def __init__(self, *args, low_clamp=-6, high_clamp=3, **kwargs):
        super().__init__(4, *args, **kwargs)
        self.low_clamp = low_clamp
        self.high_clamp = high_clamp

    def _get_dist(self, params):
        mu = params[:, :, :, :2].sigmoid()
        sigma = params[:, :, :, 2:].clamp(self.low_clamp, self.high_clamp).exp().sqrt()
        assert not mu.isnan().any() and not mu.isinf().any(), "mu contains invalid values"
        assert not sigma.isnan().any() and not sigma.isinf().any(), "sigma contains invalid values"
        return TruncatedDiagonalMVN(mu, sigma)


class LogNormalFactor(VariationalFactor):
    def __init__(self, *args, dim=1, **kwargs):
        self.dim = dim  # the dimension of a multivariate lognormal
        n_params = 2 * dim  # mean and std for each dimension (diagonal covariance)
        super().__init__(n_params, *args, **kwargs)

    def _get_dist(self, params):
        mu = params[:, :, :, 0 : self.dim].clamp(-40, 40)
        sigma = params[:, :, :, self.dim : self.n_params].clamp(-6, 5).exp().sqrt()
        iid_dist = LogNormalEpsilon(
            mu, sigma, validate_args=False
        )  # may evaluate at 0 for masked tiles
        return Independent(iid_dist, 1)


class LogitNormalFactor(VariationalFactor):
    def __init__(self, *args, low=0, high=1, dim=1, **kwargs):
        self.dim = dim  # the dimension of a multivariate logitnormal
        n_params = 2 * dim
        self.low = low
        self.high = high
        super().__init__(n_params, *args, **kwargs)

    def _get_dist(self, params):
        mu = params[:, :, :, 0 : self.dim]
        sigma = params[:, :, :, self.dim : self.n_params].clamp(-10, 10).exp().sqrt()
        return RescaledLogitNormal(mu, sigma, low=self.low, high=self.high)


#####################


class TruncatedDiagonalMVN(Distribution):
    """A truncated diagonal multivariate normal distribution."""

    def __init__(self, mu, sigma):
        """Initialize a truncated diagonal multivariate normal distribution.

        Args:
            mu (Tensor): Mean of the distribution (must be at least 1d)
            sigma (Tensor): Standard deviation of the distribution (must be at least 1d)

        Distribution is "multivariate" in that the last dimension of mu and sigma
        are considered event dimensions.
        """

        super().__init__(validate_args=False)
        multiple_normals = Normal(mu, sigma)  # all dims are batch dims, none are event
        self.base_dist = Independent(multiple_normals, 1)  # now last dim is event dim

        # we'll need these calculations later for log_prob
        prob_in_unit_box_hw = multiple_normals.cdf(self.b) - multiple_normals.cdf(self.a)
        self.log_prob_in_unit_box = prob_in_unit_box_hw.log().sum(dim=-1)

    def __repr__(self):
        return f"{self.__class__.__name__}({self.base_dist.base_dist})"

    def sample(self, sample_shape=()):
        """Generate sample.

        Args:
            sample_shape (Tuple): Shape of samples to draw

        Returns:
            Tensor: (sample_shape, self.batch_shape, self.event_shape) shaped sample

        """

        shape = sample_shape + self.base_dist.batch_shape + self.base_dist.event_shape

        # draw using inverse cdf method
        # if Fi is the cdf of the relavant gaussian, then
        # Gi(u) = Fi(u*(F(b) - F(a)) + F(a)) is the cdf of the truncated gaussian
        uniform01_samples = torch.rand(shape, device=self.base_dist.mean.device)
        uniform_fafb = uniform01_samples * (self.upper_cdf - self.lower_cdf) + self.lower_cdf
        trunc_normal_samples = self.base_dist.base_dist.icdf(uniform_fafb)

        # if u_transformed is within machine precision of 0 or 1
        # the icdf will be -inf or inf, respectively, so we have to clamp
        return trunc_normal_samples.clamp(self.a, self.b)

    @property
    def a(self):
        return torch.zeros_like(self.base_dist.mean)

    @property
    def b(self):
        return torch.ones_like(self.base_dist.mean)

    @property
    def lower_cdf(self):
        return self.base_dist.base_dist.cdf(self.a)

    @property
    def upper_cdf(self):
        return self.base_dist.base_dist.cdf(self.b)

    @property
    def mean(self):
        mu = self.base_dist.mean
        offset = self.base_dist.log_prob(self.a).exp() - self.base_dist.log_prob(self.b).exp()
        offset /= self.log_prob_in_unit_box.exp()
        return mu + (offset.unsqueeze(-1) * self.base_dist.stddev)

    @property
    def stddev(self):
        # See https://arxiv.org/pdf/1206.5387.pdf for the formula for the variance of a truncated
        # multivariate normal. The covariance terms simplify since our dimensions are independent,
        # but it's still tricky to compute.
        raise NotImplementedError("Standard deviation for truncated normal is not implemented yet")

    @property
    def mode(self):
        # a mode still exists if this assertion is false, but I haven't implemented code
        # to compute it because I don't think we need it
        assert (self.base_dist.mean >= 0).all() and (self.base_dist.mean <= 1).all()
        return self.base_dist.mode

    def log_prob(self, value):
        assert (value >= 0).all() and (value <= 1).all()
        # subtracting log probability that the base RV is in the unit box
        # is equivalent in log space to dividing the normal pdf by the normalizing constant
        return self.base_dist.log_prob(value) - self.log_prob_in_unit_box

    def cdf(self, value):
        cdf_at_val = self.base_dist.base_dist.cdf(value)
        cdf_at_lb = self.lower_cdf
        log_cdf = (cdf_at_val - cdf_at_lb + 1e-9).log().sum(dim=-1) - self.log_prob_in_unit_box
        return log_cdf.exp()


class RescaledLogitNormal(Distribution):
    def __init__(self, mu, sigma, low=0, high=1):
        super().__init__(validate_args=False)

        self.low = low
        self.high = high

        self.mu = mu
        self.sigma = sigma

        base_dist = Normal(mu, sigma)
        transforms = [SigmoidTransform(), AffineTransform(loc=self.low, scale=self.high)]
        self.iid_dist = TransformedDistribution(base_dist, transforms)

    def sample(self):
        return self.iid_dist.sample()

    def log_prob(self, value):
        return self.iid_dist.log_prob(value).sum(-1)

    @property
    def mode(self):
        # this is actual the median, not the mode. The median is suitable as a point estimate
        # and the mode doesn't have an analytic form.
        return self.iid_dist.icdf(torch.tensor(0.5))


empty_shape = torch.Size([])


class LogNormalEpsilon(LogNormal):
    def sample(self, sample_shape=empty_shape):
        sample = super().sample(sample_shape)
        max_value = torch.finfo(sample.dtype).max
        return sample.clamp(1e-9, max_value)

    def log_prob(self, value):
        return super().log_prob(value + 1e-9)
