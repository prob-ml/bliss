import torch
from torch.distributions import (
    AffineTransform,
    Categorical,
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


class UnconstrainedDiagonalBivariateNormal:
    def __init__(self):
        self.dim = 4

    def get_dist(self, params):
        mean = params[:, :, :, :2].sigmoid()
        sd = params[:, :, :, 2:].clamp(-6, 3).exp().sqrt()
        base_dist = Normal(mean, sd)
        return Independent(base_dist, 1)


class UnconstrainedLogNormal:
    def __init__(self):
        self.dim = 2

    def get_dist(self, params):
        mu = params[:, :, :, 0]
        sigma = params[:, :, :, 1].clamp(-6, 10).exp().sqrt()
        return LogNormal(mu, sigma)


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
