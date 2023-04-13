import torch
from torch.distributions import Categorical, LogNormal, Normal


class TransformedBernoulli:
    def __init__(self):
        self.dim = 1

    def get_dist(self, untransformed_params):
        yes_prob = untransformed_params.sigmoid().clamp(1e-4, 1 - 1e-4)
        no_yes_prob = torch.cat([1 - yes_prob, yes_prob], dim=3)
        return Categorical(no_yes_prob)


class TransformedNormal:
    def __init__(self, low_clamp=-20, high_clamp=20):
        self.dim = 2
        self.low_clamp = low_clamp
        self.high_clamp = high_clamp

    def get_dist(self, untransformed_params):
        mean = untransformed_params[:, :, :, 0]
        sd = untransformed_params[:, :, :, 1].clamp(self.low_clamp, self.high_clamp).exp().sqrt()
        return Normal(mean, sd)


class TransformedDiagonalBivariateNormal:
    def __init__(self):
        self.dim = 4

    def get_dist(self, untransformed_params):
        mean = untransformed_params[:, :, :, :2]
        sd = untransformed_params[:, :, :, 2:].clamp(-6, 3).exp().sqrt()
        return Normal(mean, sd)


class TransformedLogNormal:
    def __init__(self):
        self.dim = 2

    def get_dist(self, untransformed_params):
        mu = untransformed_params[:, :, :, 0]
        sigma = untransformed_params[:, :, :, 1].clamp(-6, 10).exp().sqrt()
        return LogNormal(mu, sigma)


class TransformedLogitNormal:
    def __init__(self, low=0, high=1):
        self.dim = 2
        self.low = low
        self.high = high

    def get_dist(self, untransformed_params):
        mu = untransformed_params[:, :, :, 0]
        sigma = untransformed_params[:, :, :, 1].clamp(-6, 10).exp().sqrt()
        return Normal(mu, sigma)
