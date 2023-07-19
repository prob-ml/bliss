import torch
from torch.distributions import Normal

from bliss.unconstrained_dists import TruncatedDiagonalMVN


class TestUnconstainedDists:
    def test_tdbn_univariate(self, cfg):
        # first check that a univariate normal distribution with neglible probablity mass
        # outside of the unit interval is equivalent to a truncated mvn with the same parameters
        mu = torch.ones(1) * 0.6
        sigma = torch.ones(1) * 1e-3
        tdmn = TruncatedDiagonalMVN(mu, sigma)
        claimed = tdmn.log_prob(mu)
        approx_true = Normal(mu, sigma).log_prob(mu)
        assert torch.isclose(claimed, approx_true)

        # a standard normal has about 68% of its mass in [-1,1], and therefore 34% in [0, 1].
        # check that its density function is therefore 34% of our truncated standard normal
        mu = torch.zeros(1)
        sigma = torch.ones(1)
        tdmn = TruncatedDiagonalMVN(mu, sigma)
        x = torch.ones(1) * 0.1234
        claimed = 0.34 * tdmn.log_prob(x).exp()
        approx_true = Normal(mu, sigma).log_prob(x).exp()
        assert torch.isclose(claimed, approx_true, atol=0.01)

    def test_tdbn_multivariate(self, cfg):
        # first check that a bivariate normal distribution with neglible probablity mass
        # outside of the unit box is equivalent to a truncated mvn with the same parameters
        mu = torch.ones(32, 2) * 0.6
        sigma = torch.ones(32, 2) * 1e-3
        tdmn = TruncatedDiagonalMVN(mu, sigma)
        claimed = tdmn.log_prob(mu)
        approx_true = Normal(mu, sigma).log_prob(mu).sum(-1)
        assert torch.isclose(claimed, approx_true).all()
