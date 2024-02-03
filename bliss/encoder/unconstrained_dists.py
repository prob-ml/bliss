import torch
from torch.distributions import (
    AffineTransform,
    Categorical,
    Distribution,
    LogNormal,
    Normal,
    SigmoidTransform,
    TransformedDistribution,
)


class UnconstrainedBernoulli:
    def __init__(self):
        self.dim = 1

    def get_dist(self, params):
        yes_prob = params.sigmoid().clamp(1e-4, 1 - 1e-4)
        no_yes_prob = torch.cat([1 - yes_prob, yes_prob], dim=3)
        return Categorical(no_yes_prob)


class UnconstrainedNormal:
    def __init__(self, low_clamp=-20, high_clamp=20):
        self.dim = 2
        self.low_clamp = low_clamp
        self.high_clamp = high_clamp

    def get_dist(self, params):
        mean = params[:, :, :, 0]
        sd = params[:, :, :, 1].clamp(self.low_clamp, self.high_clamp).exp().sqrt()
        return Normal(mean, sd)


class TruncatedDiagonalMVN(Distribution):
    """A truncated diagonal multivariate normal distribution."""

    def __init__(self, mu, sigma, a, b):
        super().__init__(validate_args=False)

        self.dim = mu.size()

        self.lb = a * torch.ones_like(mu)
        self.ub = b * torch.ones_like(mu)

        base = Normal(mu, sigma)
        prob_in_box_hw = base.cdf(b * torch.ones_like(mu)) - base.cdf(a * torch.ones_like(mu))
        self.log_prob_in_box = prob_in_box_hw.log()

        self.base_dist = base

    def __repr__(self):
        return f"{self.__class__.__name__}({self.base_dist})"

    def sample(self, **args):
        p = torch.rand(tuple(self.dim)).clamp(min=1e-6, max=1.0 - 1e-6)
        p_tilde = self.base_dist.cdf(self.lb) + p * (self.log_prob_in_box.exp())

        return self.base_dist.icdf(p_tilde)

    @property
    def mean(self):
        mu = self.base_dist.mean
        offset = self.base_dist.log_prob(self.lb).exp() - self.base_dist.log_prob(self.ub).exp()
        offset /= self.log_prob_in_box.exp()
        return mu + (offset * self.base_dist.stddev)

    @property
    def mode(self):
        assert (self.mean >= self.lb).all() and (self.mean <= self.ub).all()
        return self.base_dist.mode

    def log_prob(self, value):
        assert (value >= self.lb).all() and (value <= self.ub).all()
        return self.base_dist.log_prob(value) - self.log_prob_in_box

    def cdf(self, value):
        cdf_at_val = self.base_dist.cdf(value)
        cdf_at_lb = self.base_dist.cdf(self.lb * torch.ones_like(self.mean))
        log_cdf = (cdf_at_val - cdf_at_lb + 1e-9).log().sum(dim=-1) - self.log_prob_in_box
        return log_cdf.exp()


class UnconstrainedTDBN:
    """Produces truncated bivariate normal distributions from unconstrained parameters."""

    def __init__(self):
        self.dim = 4

    def get_dist(self, params):
        mu = params[:, :, :, :2].tanh()
        sigma = params[:, :, :, 2:].clamp(-6, 3).exp().sqrt()
        return TruncatedDiagonalMVN(mu, sigma, a=-1, b=1)


class UnconstrainedLogNormal:
    def __init__(self):
        self.dim = 2

    def get_dist(self, params):
        mu = params[:, :, :, 0]
        sigma = params[:, :, :, 1].clamp(-6, 10).exp().sqrt()
        return LogNormal(mu, sigma, validate_args=False)  # we may evaluate at 0 for masked tiles


class UnconstrainedLogitNormal:
    def __init__(self, low=0, high=1):
        self.dim = 2
        self.low = low
        self.high = high

    def get_dist(self, params):
        mu = params[:, :, :, 0]
        sigma = params[:, :, :, 1].clamp(-10, 10).exp().sqrt()
        base_dist = Normal(mu, sigma)
        transforms = [SigmoidTransform(), AffineTransform(loc=self.low, scale=self.high)]
        return TransformedDistribution(base_dist, transforms)
