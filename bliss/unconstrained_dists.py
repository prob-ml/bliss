import torch
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

    def __init__(self, mu, sigma):
        super().__init__()

        multiple_normals = Normal(mu, sigma)
        prob_in_unit_box_hw = multiple_normals.cdf(torch.ones_like(mu))
        prob_in_unit_box_hw -= multiple_normals.cdf(torch.zeros_like(mu))
        self.log_prob_in_unit_box = prob_in_unit_box_hw.log().sum(dim=-1)

        self.base_dist = Independent(multiple_normals, 1)

    def sample(self, **args):
        # some ideas for how to sample it here, if we need to:
        # https://cran.r-project.org/web/packages/truncnorm/
        raise NotImplementedError("sampling a truncated normal isn't straightforward")

    @property
    def mode(self):
        mu = self.base_dist.mean
        # a mode still exists if this assertion is false, but I haven't implemented code
        # to compute it because I don't think we need it
        assert (mu >= 0).all() and (mu <= 1).all()
        return self.base_dist.mode

    def log_prob(self, value):
        assert (value >= 0).all() and (value <= 1).all()
        # subtracting log probability that the base RV is in the unit box
        # is equivalent in log space to dividing the normal pdf by the normalizing constant
        return self.base_dist.log_prob(value) - self.log_prob_in_unit_box


class UnconstrainedTDBN:
    """Produces truncated bivariate normal distributions from unconstrained parameters."""

    def __init__(self):
        self.dim = 4

    def get_dist(self, params):
        mu = params[:, :, :, :2].sigmoid()
        sigma = params[:, :, :, 2:].clamp(-6, 3).exp().sqrt()
        return TruncatedDiagonalMVN(mu, sigma)


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
