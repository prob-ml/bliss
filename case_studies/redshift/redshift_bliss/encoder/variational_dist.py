import einops
import torch
from einops import rearrange
from torch.distributions import Categorical, Distribution, MixtureSameFamily, Normal

from bliss.catalog import BaseTileCatalog
from bliss.encoder.variational_dist import VariationalDist, VariationalFactor
from case_studies.redshift.splinex.bspline import DeclanBSpline


class NormalFactor(VariationalFactor):
    def __init__(self, *args, low_clamp=-20, high_clamp=20, **kwargs):
        super().__init__(2, *args, **kwargs)
        self.low_clamp = low_clamp
        self.high_clamp = high_clamp

    def get_dist(self, params):
        mean = params[:, :, :, 0:1]
        sd = params[:, :, :, 1:2].clamp(self.low_clamp, self.high_clamp).exp().sqrt()
        return Normal(mean, sd)

    def get_mode(self, params):
        qk = self.get_dist(params)
        mode = qk.mean
        if self.sample_rearrange is not None:
            mode = rearrange(mode, self.sample_rearrange)
            assert mode.isfinite().all(), f"mode has invalid values: {mode}"
        return mode

    def get_mean(self, params):
        return self.get_mode(params)

    def get_median(self, params):
        return self.get_mode(params)


class DiscretizedFactor1D(VariationalFactor):
    def __init__(self, n_params, *args, low=0, high=3, **kwargs):
        super().__init__(n_params, *args, **kwargs)
        self.low = low
        self.high = high

    def get_dist(self, params):
        return Discretized1D(params, self.low, self.high, self.n_params)

    def discrete_sample(
        self, params, use_mode=False, risk_type="redshift_outlier_fraction_catastrophic_bin"
    ):
        qk = self.get_dist(params)
        sample_cat = qk.get_lowest_risk_bin(risk_type=risk_type)
        if self.sample_rearrange is not None:
            sample_cat = rearrange(sample_cat, self.sample_rearrange)
            assert sample_cat.isfinite().all(), f"sample_cat has invalid values: {sample_cat}"
        return sample_cat

    def get_mode(self, params):
        qk = self.get_dist(params)
        mode = qk.mode
        if self.sample_rearrange is not None:
            mode = rearrange(mode, self.sample_rearrange)
            assert mode.isfinite().all(), f"mode has invalid values: {mode}"
        return mode

    def get_mean(self, params):
        qk = self.get_dist(params)
        mean = qk.mean
        if self.sample_rearrange is not None:
            mean = rearrange(mean, self.sample_rearrange)
            assert mean.isfinite().all(), f"mean has invalid values: {mean}"
        return mean

    def get_median(self, params):
        qk = self.get_dist(params)
        median = qk.median
        if self.sample_rearrange is not None:
            median = rearrange(median, self.sample_rearrange)
            assert median.isfinite().all(), f"median has invalid values: {median}"
        return median


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

    @property
    def median(self):
        """Compute the median of the distribution.

        Median is defined as the midpoint of the first bin
        where the cumulative distribution function (CDF) is >= 0.5.
        """
        cum_probs = self.base_dist.probs.cumsum(dim=-1)
        median_index = (cum_probs >= 0.5).float().argmax(dim=-1)
        median = (median_index.float() + 0.5) / self.num_bins * (self.high - self.low) + self.low
        return median

    @property
    def mean(self):
        """Compute the mean of the distribution.
        Mean is computed as the weighted average of the bin centers,
        where the weights are the probabilities of each bin.
        """
        bin_centers = torch.arange(
            self.low + 0.5 * (self.high - self.low) / self.num_bins,
            self.high - 0.5 * (self.high - self.low) / self.num_bins,
            (self.high - self.low) / self.num_bins,
            device=self.base_dist.logits.device,
        )
        bin_probs = self.base_dist.probs
        res = bin_probs * bin_centers
        mean = einops.reduce(res, "b l w n_bins -> b l w", "sum")
        return mean

    def log_prob(self, value):
        locs = ((value - self.low) / (self.high - self.low)).clamp(0, 1 - 1e-7)
        locbins = (locs * self.num_bins).long()
        return self.base_dist.log_prob(locbins) + self.pdf_offset

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

    # noqa: WPS223
    def get_lowest_risk_bin(self, risk_type="redshift_outlier_fraction_catastrophic_bin"):
        RISK_FUNCTIONS = {
            "redshift_outlier_fraction_catastrophic_bin_mag": self.compute_catastrophic_risk,
            "redshift_outlier_fraction_catastrophic_bin_rs": self.compute_catastrophic_risk,
            "redshift_outlier_fraction_bin_mag": self.compute_outlier_fraction_risk,
            "redshift_outlier_fraction_bin_rs": self.compute_outlier_fraction_risk,
            "redshift_nmad_bin_mag": self.compute_nmad_risk,
            "redshift_nmad_bin_rs": self.compute_nmad_risk,
            "redshift_mean_square_error_bin_mag": self.compute_mse_risk,
            "redshift_mean_square_error_bin_rs": self.compute_mse_risk,
            "redshift_abs_bias_bin_mag": self.compute_abs_bias_risk,
            "redshift_abs_bias_bin_rs": self.compute_abs_bias_risk,
            "redshift_L1_bin_mag": self.compute_abs_bias_risk,
            "redshift_L1_bin_rs": self.compute_abs_bias_risk,
        }

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
            try:
                risk_fn = RISK_FUNCTIONS[risk_type]
            except KeyError as exc:
                raise ValueError(f"Invalid risk type: {risk_type}") from exc

            risk = risk_fn(z_pred, bin_centers, bin_probs)

            update_mask = risk < min_risk
            min_risk = torch.where(update_mask, risk, min_risk)
            best_bin = torch.where(update_mask, z_pred, best_bin)

        return best_bin


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

    def get_mode(self, x_cat):
        fp_pairs = self._factor_param_pairs(x_cat)
        d = {qk.name: qk.get_mode(params) for qk, params in fp_pairs}
        return BaseTileCatalog(d)

    def get_mean(self, x_cat):
        fp_pairs = self._factor_param_pairs(x_cat)
        d = {qk.name: qk.get_mean(params) for qk, params in fp_pairs}
        return BaseTileCatalog(d)

    def get_median(self, x_cat):
        fp_pairs = self._factor_param_pairs(x_cat)
        d = {qk.name: qk.get_median(params) for qk, params in fp_pairs}
        return BaseTileCatalog(d)


class AbstractSplineClass:
    _cache = {}

    def __init__(self, min_val, max_val, num_knots, n_grid_points, degree, device):
        self.min_val = min_val
        self.max_val = max_val
        self.num_knots = num_knots
        self.n_grid_points = n_grid_points
        self.degree = degree
        self.device = device

    def _create_spline(self):
        spline = self._cache.get("spline")
        if spline is None:
            spline = DeclanBSpline(
                min_val=self.min_val,
                max_val=self.max_val,
                nknots=self.num_knots,
                n_grid_points=self.n_grid_points,
                degree=self.degree,
                device=self.device,
            )
            self._cache["spline"] = spline

        return spline


class BSpline1D(Distribution):
    """A continuous bivariate distribution over the 2d unit box, with a discretized density."""

    def __init__(self, coeffs, min_val=0, max_val=3, num_knots=40, n_grid_points=250, degree=3):
        super().__init__(validate_args=False)
        self.min_val = min_val
        self.max_val = max_val
        self.num_knots = num_knots
        self.n_grid_points = n_grid_points
        self.degree = degree
        device = coeffs.device
        self.abstract_spline = AbstractSplineClass(
            min_val=min_val,
            max_val=max_val,
            num_knots=num_knots,
            n_grid_points=n_grid_points,
            degree=degree,
            device=device,
        )
        self.spline = self.abstract_spline._create_spline()
        self.coeffs = coeffs
        self.grid = self.spline.t_values.to(self.coeffs.device)
        self.pdf = self._pdf()  # grid_size b l w
        self.delta = self.grid[1] - self.grid[0]

    def _pdf(self):
        r = einops.rearrange(self.coeffs, "b l w coeff -> coeff (b l w)")
        spline_curve_vals, _, _ = self.spline(r)
        spline_curve_vals = spline_curve_vals.clamp(min=-10.0, max=10.0)
        exped = torch.exp(spline_curve_vals)
        t_values = self.spline.t_values.to(self.coeffs.device)
        integrals = torch.trapezoid(y=exped, x=t_values, axis=0)
        exped = exped / integrals.unsqueeze(0)
        exped = einops.rearrange(
            exped,
            "n_grid_pts (b l w) -> b l w n_grid_pts",
            b=self.coeffs.shape[0],
            l=self.coeffs.shape[1],
            w=self.coeffs.shape[2],
        )

        return exped

    def sample(self, sample_shape=()):
        raise NotImplementedError("Sampling from B-spline is not implemented yet.")

    @property
    def mode(self):
        pdf = self.pdf
        max_index = pdf.argmax(dim=-1)
        modes = self.grid[max_index]
        return modes

    @property
    def median(self):
        return self.quantile(0.5)

    @property
    def mean(self):
        riemann = (
            self.pdf * einops.rearrange(self.grid, "n_grid_pts -> 1 1 1 n_grid_pts") * self.delta
        )
        mean = einops.reduce(riemann, "b l w n_grid_pts -> b l w", "sum")
        return mean

    def one_hot_nearest_index(self, grid: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        diff = torch.abs(grid[:, None] - values[None, :])  # (len(grid), len(values))
        closest_indices = torch.argmin(diff, dim=0)  # Shape: (len(values),)
        one_hot = torch.nn.functional.one_hot(
            closest_indices, num_classes=grid.shape[0]
        )  # Shape: (len(values), len(grid))
        return one_hot

    def log_prob(self, value):
        r = einops.rearrange(self.coeffs, "b l w coeff -> coeff (b l w)")
        # mask = (value != 0).float()
        # mask = einops.rearrange(mask, 'b l w -> (b l w)')
        t_values = self.spline.t_values.to(self.coeffs.device)
        theta = einops.rearrange(value, "b l w -> (b l w)")
        spline_curve_vals, _, _ = self.spline(r)
        spline_curve_vals = spline_curve_vals.clamp(min=-10.0, max=10.0)
        exped = torch.exp(spline_curve_vals)

        integrals = torch.trapezoid(y=exped, x=t_values, axis=0)

        one_hotted = self.one_hot_nearest_index(t_values, theta)
        nums = torch.mul(one_hotted, spline_curve_vals.T).sum(-1)
        lps = nums - torch.log(integrals)
        lps = einops.rearrange(
            lps, "(b l w) -> b l w", b=value.shape[0], l=value.shape[1], w=value.shape[2]
        )
        return lps

    def _cdf(self):
        # Use truncated CDF
        cdf_values = torch.cumsum(
            self.pdf * self.delta, dim=-1
        )  # TODO: should this be a left or right Riemann sum?
        trunc_cdf_values = (cdf_values - cdf_values[..., 0].unsqueeze(-1)) / (
            cdf_values[..., -1] - cdf_values[..., 0]
        ).unsqueeze(-1)
        return trunc_cdf_values

    def cdf(self, value):
        if not isinstance(value, torch.Tensor):
            value = torch.tensor([value])
        min_val = self.grid[0]
        max_val = self.grid[-1]
        value = torch.clamp(value, min=min_val, max=max_val)
        index_to_use = torch.abs(self.grid - value).argmin()
        cdf = self._cdf()  # grid_size b l w
        return cdf[..., index_to_use]  # b l w

    def quantile(self, prob_values):
        if not isinstance(prob_values, torch.Tensor):
            prob_values = torch.tensor([prob_values])
        prob_values = prob_values.to(self.grid.device)
        prob_values = torch.clamp(prob_values, min=0.0, max=1.0)
        quantiles = []
        cdf_values = self._cdf()
        grid_points = self.grid

        for _, p in enumerate(prob_values):
            idx = (cdf_values - p >= 0).float().argmax(-1)
            idx = idx.clamp(1, len(self.grid) - 1).long()

            x1, x2 = grid_points[idx - 1], grid_points[idx]
            indexed_cdf = cdf_values
            idx_expanded = idx.unsqueeze(-1)

            y1 = torch.gather(indexed_cdf, dim=-1, index=idx_expanded - 1)
            y2 = torch.gather(indexed_cdf, dim=-1, index=idx_expanded)
            y1 = einops.rearrange(y1, "b l w 1 -> b l w")
            y2 = einops.rearrange(y2, "b l w 1 -> b l w")
            res = x1 + (p - y1) * (x2 - x1) / (y2 - y1)
            quantiles.append(res)

        if len(quantiles) == 1:
            quantiles = quantiles[0]
        else:
            quantiles = torch.stack(quantiles, dim=0)
        return quantiles


class BSplineFactor1D(VariationalFactor):
    def __init__(
        self, *args, min_val=0, max_val=3, num_knots=40, n_grid_points=250, degree=3, **kwargs
    ):
        super().__init__(num_knots, *args, **kwargs)  # num_knots is the number of params
        self.min_val = min_val
        self.max_val = max_val
        self.num_knots = num_knots
        self.n_grid_points = n_grid_points
        self.degree = degree

    def get_dist(self, params):
        return BSpline1D(
            params, self.min_val, self.max_val, self.num_knots, self.n_grid_points, self.degree
        )

    def discrete_sample(self):
        raise NotImplementedError("Sampling from B-spline is not implemented yet.")

    def get_mode(self, params):
        qk = self.get_dist(params)
        mode = qk.mode
        if self.sample_rearrange is not None:
            mode = rearrange(mode, self.sample_rearrange)
            assert mode.isfinite().all(), f"mode has invalid values: {mode}"
        return mode

    def get_mean(self, params):
        qk = self.get_dist(params)
        mean = qk.mean
        if self.sample_rearrange is not None:
            mean = rearrange(mean, self.sample_rearrange)
            assert mean.isfinite().all(), f"mean has invalid values: {mean}"
        return mean

    def get_median(self, params):
        qk = self.get_dist(params)
        median = qk.median
        if self.sample_rearrange is not None:
            median = rearrange(median, self.sample_rearrange)
            assert median.isfinite().all(), f"median has invalid values: {median}"
        return median


class MixtureOfGaussians1D(Distribution):
    """A mixture of Gaussians distribution in 1D."""

    def __init__(self, params, n_comp=5, min_val=0, max_val=3):
        assert n_comp == params.shape[-1] // 3
        super().__init__(validate_args=False)
        self.n_comp = n_comp
        self.means = params[..., :n_comp]
        self.log_stds = params[..., n_comp : 2 * n_comp].clamp(-20.0, 20.0)
        self.weights = params[..., 2 * n_comp :]
        self.weights = torch.nn.functional.softmax(self.weights, dim=-1)
        self.stds = torch.exp(self.log_stds)
        mix = Categorical(probs=self.weights)
        comp = Normal(self.means, self.stds)
        self.mixture = MixtureSameFamily(mix, comp)
        self.grid = torch.linspace(min_val, max_val, 1000, device=params.device)
        self.rgrid = einops.repeat(
            self.grid, "g -> g b l w", b=params.shape[0], l=params.shape[1], w=params.shape[2]
        )
        self.pdf = self.mixture.log_prob(self.rgrid).exp()
        self.delta = self.grid[1] - self.grid[0]

    def sample(self, sample_shape=()):
        raise NotImplementedError("Sampling from MDN is not implemented yet.")

    @property
    def mode(self):
        to_select = self.pdf.argmax(dim=0)
        selected = self.grid[to_select]
        return selected

    @property
    def median(self):
        return self.quantile(0.5)

    @property
    def mean(self):
        # Mean is weighted averages of the means of the components
        dotted = self.means * self.weights
        mean = einops.reduce(dotted, "b l w n_comp -> b l w", "sum")
        return mean

    def log_prob(self, value):
        return self.mixture.log_prob(value)

    def _cdf(self):
        # Use truncated CDF
        cdf_values = torch.cumsum(
            self.pdf * self.delta, dim=0
        )  # TODO: should this be a left or right Riemann sum?
        trunc_cdf_values = (cdf_values - cdf_values[0].unsqueeze(0)) / (
            cdf_values[-1] - cdf_values[0]
        ).unsqueeze(0)
        return trunc_cdf_values

    def cdf(self, value):
        if not isinstance(value, torch.Tensor):
            value = torch.tensor([value])
        min_val = self.grid[0]
        max_val = self.grid[-1]
        value = torch.clamp(value, min=min_val, max=max_val)
        index_to_use = torch.abs(self.grid - value).argmin()
        cdf = self._cdf()  # grid_size b l w
        return cdf[index_to_use]  # b l w

    def quantile(self, prob_values):
        if not isinstance(prob_values, torch.Tensor):
            prob_values = torch.tensor([prob_values])
        prob_values = prob_values.to(self.grid.device)
        prob_values = torch.clamp(prob_values, min=0.0, max=1.0)
        quantiles = []
        cdf_values = self._cdf()
        grid_points = self.grid

        for _, p in enumerate(prob_values):
            idx = (cdf_values - p >= 0).float().argmax(0)
            idx = idx.clamp(1, len(cdf_values) - 1).long()

            x1, x2 = grid_points[idx - 1], grid_points[idx]
            indexed_cdf = einops.rearrange(cdf_values, "ngrid b l w -> b l w ngrid")
            idx_expanded = idx.unsqueeze(-1)

            y1 = torch.gather(indexed_cdf, dim=-1, index=idx_expanded - 1)
            y2 = torch.gather(indexed_cdf, dim=-1, index=idx_expanded)
            y1 = einops.rearrange(y1, "b l w 1 -> b l w")
            y2 = einops.rearrange(y2, "b l w 1 -> b l w")
            res = x1 + (p - y1) * (x2 - x1) / (y2 - y1)
            quantiles.append(res)

        if len(quantiles) == 1:
            quantiles = quantiles[0]
        else:
            quantiles = torch.stack(quantiles, dim=0)
        return quantiles


class MixtureOfGaussiansFactor1D(VariationalFactor):
    def __init__(self, n_comp, *args, **kwargs):
        super().__init__(n_comp * 3, *args, **kwargs)  # num_knots is the number of params

    def get_dist(self, params):
        return MixtureOfGaussians1D(params)

    def discrete_sample(self):
        raise NotImplementedError("Sampling from MDN is not implemented yet.")

    def get_mode(self, params):
        qk = self.get_dist(params)
        mode = qk.mode
        if self.sample_rearrange is not None:
            mode = rearrange(mode, self.sample_rearrange)
            assert mode.isfinite().all(), f"mode has invalid values: {mode}"
        return mode

    def get_mean(self, params):
        qk = self.get_dist(params)
        mean = qk.mean
        if self.sample_rearrange is not None:
            mean = rearrange(mean, self.sample_rearrange)
            assert mean.isfinite().all(), f"mean has invalid values: {mean}"
        return mean

    def get_median(self, params):
        qk = self.get_dist(params)
        median = qk.median
        if self.sample_rearrange is not None:
            median = rearrange(median, self.sample_rearrange)
            assert median.isfinite().all(), f"median has invalid values: {median}"
        return median
