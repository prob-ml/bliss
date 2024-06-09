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
        # this next line may be helpful with nans encountered during training with fp16s
        no_yes_prob = no_yes_prob.nan_to_num(nan=0.5)
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


class UnconstrainedBivariateNormal:
    def __init__(self, low_clamp=-20, high_clamp=20):
        self.dim = 4
        self.low_clamp = low_clamp
        self.high_clamp = high_clamp

    def get_dist(self, params):
        mean = params[:, :, :, :2]
        sd = params[:, :, :, 2:].clamp(self.low_clamp, self.high_clamp).exp().sqrt()

        return Independent(Normal(mean, sd), 1)


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
        uniform_samples = torch.rand(shape, device=self.base_dist.mean.device)

        with torch.no_grad():
            # draw using inverse cdf method
            # if Fi is the cdf of the relavant gaussian, then
            # Gi(u) = Fi(u*(F(b) - F(a)) + F(a)) is the cdf of the truncated gaussian
            # if u_transformed is within machine precision of 0 or 1
            # the icdf will be -inf or inf, respectively, so we have to clamp
            return self.base_dist.base_dist.icdf(
                uniform_samples * (self.upper_cdf - self.lower_cdf) + self.lower_cdf,
            ).clamp(self.a, self.b)

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


class UnconstrainedTDBN:
    """Produces truncated bivariate normal distributions from unconstrained parameters."""

    def __init__(self, low_clamp=-6, high_clamp=3):
        self.dim = 4
        self.low_clamp = low_clamp
        self.high_clamp = high_clamp

    def get_dist(self, params):
        mu = params[:, :, :, :2].sigmoid()
        sigma = params[:, :, :, 2:].clamp(self.low_clamp, self.high_clamp).exp().sqrt()
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
