import torch
from einops import rearrange
from torch.distributions import Categorical, Distribution

from bliss.catalog import BaseTileCatalog
from bliss.encoder.variational_dist import VariationalDist, VariationalFactor


class DiscretizedFactor1D(VariationalFactor):
    def __init__(self, n_params, low=0, high=3, *args, **kwargs):
        super().__init__(n_params, *args, **kwargs)
        self.low = low
        self.high = high

    def _get_dist(self, params):
        return Discretized1D(params, self.low, self.high, self.n_params)

    def discrete_sample(
        self, params, use_mode=False, risk_type="redshift_outlier_fraction_catastrophic_bin"
    ):
        qk = self._get_dist(params)
        sample_cat = qk.get_lowest_risk_bin(risk_type=risk_type)
        if self.sample_rearrange is not None:
            sample_cat = rearrange(sample_cat, self.sample_rearrange)
            assert sample_cat.isfinite().all(), f"sample_cat has invalid values: {sample_cat}"
        return sample_cat


#####################


class Discretized1D(Distribution):
    """A continuous bivariate distribution over the 2d unit box, with a discretized density."""

    def __init__(self, logits, low=0, high=3, num_bins=30):
        super().__init__(validate_args=False)
        self.low = low
        self.high = high
        self.num_bins = num_bins
        assert logits.shape[-1] == self.num_bins
        self.pdf_offset = torch.tensor(
            num_bins / (self.high - self.low), device=logits.device
        ).log()
        self.base_dist = Categorical(logits=logits)

    def sample(self, sample_shape=()):
        locbins = self.base_dist.sample(sample_shape)
        locbins_float = locbins.float()
        return (locbins_float + torch.rand_like(locbins_float)) / self.num_bins * (
            self.high - self.low
        ) + self.low

    @property
    def mode(self):
        locbins = self.base_dist.probs.argmax(dim=-1)
        return (locbins.float() + 0.5) / self.num_bins * (self.high - self.low) + self.low

    def compute_catastrophic_risk(self, z_pred, bin_centers, bin_probs):
        """Compute the catastrophic risk for a predicted redshift (z_pred)."""
        risk = torch.zeros_like(bin_probs[..., 0])
        for i in range(self.num_bins):
            z_i = bin_centers[i]

            catastrophic_mask = torch.abs(z_pred - z_i) > 1

            if catastrophic_mask:
                risk += bin_probs[..., i]

        return risk

    def compute_outlier_fraction_risk(self, z_pred, bin_centers, bin_probs):
        """Compute the outlier fraction risk for a predicted redshift (z_pred).
        Defined via `|z_true - z_pred| / (1 + z_true) > 0.15` as outlier.

        Args:
            z_pred: Predicted redshifts
            bin_centers: Centers of the redshift bins
            bin_probs: Probability of each bin

        Returns:
            risk: catastrophic outlier frac
        """
        risk = torch.zeros_like(bin_probs[..., 0])
        for i in range(self.num_bins):
            z_i = bin_centers[i]

            outlier_mask = torch.abs(z_pred - z_i) / (1 + z_i) > 0.15

            if outlier_mask:
                risk += bin_probs[..., i]

        return risk

    def compute_nmad_risk(self, z_pred, bin_centers, bin_probs):
        """Compute the NMAD risk for a predicted redshift (z_pred).
        NMAD = `1.4826 * Median(|(z_true - z_pred) / (1 + z_true)| -`
            `Median((z_true - z_pred) / (1 + z_true)))`.

        Args:
            z_pred: Predicted redshifts
            bin_centers: Centers of the redshift bins
            bin_probs: Probability of each bin

        Returns:
            risk: nmad computation from docstring above
        """
        risk = torch.zeros_like(bin_probs[..., 0])
        for i in range(self.num_bins):
            z_i = bin_centers[i]

            abs_diff = torch.abs(z_pred - z_i) / (1 + z_i)
            median_abs_diff = torch.median(abs_diff)
            risk += bin_probs[..., i] * median_abs_diff

        return risk

    def compute_mse_risk(self, z_pred, bin_centers, bin_probs):
        """Compute the MSE risk for a predicted redshift (z_pred)."""
        risk = torch.zeros_like(bin_probs[..., 0])
        for i in range(self.num_bins):
            z_i = bin_centers[i]

            risk += bin_probs[..., i] * (z_pred - z_i) ** 2

        return risk

    def compute_abs_bias_risk(self, z_pred, bin_centers, bin_probs):
        """Compute the absolute bias risk for a predicted redshift (z_pred)."""
        risk = torch.zeros_like(bin_probs[..., 0])
        for i in range(self.num_bins):
            z_i = bin_centers[i]

            risk += bin_probs[..., i] * torch.abs(z_pred - z_i)

        return risk

    def get_lowest_risk_bin(self, risk_type="redshift_outlier_fraction_catastrophic_bin"):
        """Find the bin that minimizes the risk of producing a catastrophic outlier."""
        bin_centers = torch.linspace(
            self.low, self.high, self.num_bins, device=self.base_dist.logits.device
        )
        bin_probs = self.base_dist.probs

        min_risk = torch.full(
            bin_probs.shape[:-1], float("inf"), device=self.base_dist.logits.device
        )  # Shape [N, H, W]
        best_bin = torch.zeros_like(min_risk)  # Shape [N, H, W]

        # Grid search
        for z_pred in bin_centers:
            if risk_type == "redshift_outlier_fraction_catastrophic_bin":
                risk = self.compute_catastrophic_risk(z_pred, bin_centers, bin_probs)
            if risk_type == "redshift_outlier_fraction_bin":
                risk = self.compute_outlier_fraction_risk(z_pred, bin_centers, bin_probs)
            if risk_type == "redshift_nmad_bin":
                risk = self.compute_nmad_risk(z_pred, bin_centers, bin_probs)
            if risk_type == "redshift_mearn_square_error_bin":
                risk = self.compute_mse_risk(z_pred, bin_centers, bin_probs)
            if risk_type == "redshift_abs_bias_bin":
                risk = self.compute_abs_bias_risk(z_pred, bin_centers, bin_probs)
            else:
                raise ValueError(f"Invalid risk type: {risk_type}")

            update_mask = risk < min_risk
            min_risk = torch.where(update_mask, risk, min_risk)
            best_bin = torch.where(update_mask, z_pred, best_bin)

        return best_bin

    def log_prob(self, value):
        locs = ((value - self.low) / (self.high - self.low)).clamp(0, 1 - 1e-7)
        locbins = (locs * self.num_bins).long()
        return self.base_dist.log_prob(locbins) + self.pdf_offset


class RedshiftVariationalDist(VariationalDist):
    def discrete_sample(self, x_cat, use_mode=False, risk_type=None):
        fp_pairs = self._factor_param_pairs(x_cat)
        d = {qk.name: qk.discrete_sample(params, use_mode, risk_type) for qk, params in fp_pairs}
        return BaseTileCatalog(d)

    def sample(self, x_cat, use_mode=True):
        fp_pairs = self._factor_param_pairs(x_cat)
        d = {qk.name: qk.sample(params, use_mode) for qk, params in fp_pairs}

        # BaseTileCatalog b/c no other quantities (e.g. locs)
        return BaseTileCatalog(d)
