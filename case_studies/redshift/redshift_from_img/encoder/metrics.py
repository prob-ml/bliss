# flake8: noqa: WPS348
from typing import List, Union

import torch
from einops import rearrange
from torch import Tensor
from torchmetrics import Metric

from bliss.catalog import BaseTileCatalog
from bliss.encoder.metrics import CatalogMatcher


class MetricBin(Metric):
    def __init__(
        self,
        mag_band: int = 2,
        mag_bin_cutoffs: list = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.mag_band = mag_band
        self.mag_bin_cutoffs = mag_bin_cutoffs
        self.n_bins = len(self.mag_bin_cutoffs) + 1


class RedshiftMeanSquaredError(Metric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state("sum_squared_error", default=torch.zeros(1), dist_reduce_fx="sum")
        self.add_state("total", default=torch.zeros(1), dist_reduce_fx="sum")

    def update(self, true_cat, est_cat, matching):
        for i in range(true_cat.batch_size):
            tcat_matches, ecat_matches = matching[i]
            self.total += tcat_matches.sum()

            true_red = true_cat["redshifts"][i][tcat_matches]
            est_red = est_cat["redshifts"][i][ecat_matches]
            red_err = ((true_red - est_red).abs() ** 2).sum()

            self.sum_squared_error += red_err

    def compute(self):
        print(f"total num of pts: {self.total}")  # noqa: WPS421
        mse = self.sum_squared_error / self.total
        return {"redshifts/mse": mse.item()}


class RedshiftMeanSquaredErrorBin(MetricBin):
    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.add_state("sum_squared_error", default=torch.zeros(self.n_bins), dist_reduce_fx="sum")
        self.add_state("total", default=torch.zeros(self.n_bins), dist_reduce_fx="sum")

    def update(self, true_cat, est_cat, matching):
        cutoffs = torch.tensor(self.mag_bin_cutoffs, device=self.device)
        for i in range(true_cat.batch_size):
            tcat_matches, ecat_matches = matching[i]

            true_red = true_cat["redshifts"][i][tcat_matches].to(self.device)
            est_red = est_cat["redshifts"][i][ecat_matches].to(self.device)

            true_mag = (
                true_cat.on_nmgy[i][..., self.mag_band].unsqueeze(-1)[tcat_matches].to(self.device)
            )
            bin_indices = torch.bucketize(true_mag, cutoffs)

            red_err = (true_red - est_red).abs() ** 2

            self.total += bin_indices.bincount(minlength=self.n_bins)
            for bin_idx, err in zip(bin_indices, red_err):
                self.sum_squared_error[bin_idx] += err

    def compute(self):
        print(f"total num of pts: {self.total}")  # noqa: WPS421
        mse_per_bin = self.sum_squared_error / self.total
        mse_per_bin_results = {
            f"redshifts/mse_bin_{i}": mse_per_bin[i].item() for i in range(len(mse_per_bin))
        }
        return {**mse_per_bin_results}


class RedshiftOutlierFraction(Metric):
    """set |z_true - z_pred| / (1 + z_true) > 0.15 as outlier,  # noqa: RST305 D415
    |z_true - z_pred| > 1 as catastrophic outlier, then calculate
    fraction by outlier / total .
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state("num_outlier", default=torch.zeros(1), dist_reduce_fx="sum")
        self.add_state("total", default=torch.zeros(1), dist_reduce_fx="sum")

    def update(self, true_cat, est_cat, matching):
        for i in range(true_cat.batch_size):
            tcat_matches, ecat_matches = matching[i]
            self.total += tcat_matches.sum()

            true_red = true_cat["redshifts"][i][tcat_matches]
            est_red = est_cat["redshifts"][i][ecat_matches]
            metric_outlier = torch.abs(true_red - est_red) / (1 + true_red)
            self.num_outlier += (metric_outlier > 0.15).sum()

    def compute(self):
        outlier_fraction = self.num_outlier / self.total
        return {"redshifts/outlier_fraction": outlier_fraction.item()}


class RedshiftOutlierFractionBin(MetricBin):
    """set |z_true - z_pred| / (1 + z_true) > 0.15 as outlier,  # noqa: RST305 D415
    |z_true - z_pred| > 1 as catastrophic outlier, then calculate
    fraction by outlier / total .
    """

    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.add_state("num_outlier", default=torch.zeros(self.n_bins), dist_reduce_fx="sum")
        self.add_state("total", default=torch.zeros(self.n_bins), dist_reduce_fx="sum")

    def update(self, true_cat, est_cat, matching):
        cutoffs = torch.tensor(self.mag_bin_cutoffs, device=self.device)
        for i in range(true_cat.batch_size):
            tcat_matches, ecat_matches = matching[i]

            true_red = true_cat["redshifts"][i][tcat_matches].to(self.device)
            est_red = est_cat["redshifts"][i][ecat_matches].to(self.device)
            true_mag = (
                true_cat.on_nmgy[i][..., self.mag_band].unsqueeze(-1)[tcat_matches].to(self.device)
            )
            bin_indices = torch.bucketize(true_mag, cutoffs)

            metric_outlier = torch.abs(true_red - est_red) / (1 + true_red)
            metric_outlier = metric_outlier > 0.15

            self.total += bin_indices.bincount(minlength=self.n_bins)
            for bin_idx, outlier in zip(bin_indices, metric_outlier):
                self.num_outlier[bin_idx] += outlier

    def compute(self):
        outlier_fraction_per_bin = self.num_outlier / self.total
        outlier_fraction_per_bin_results = {
            f"redshifts/outlier_fraction_bin_{i}": outlier_fraction_per_bin[i].item()
            for i in range(len(outlier_fraction_per_bin))
        }
        return {**outlier_fraction_per_bin_results}


class RedshiftOutlierFractionCata(Metric):
    """set |z_true - z_pred| / (1 + z_true) > 0.15 as outlier,  # noqa: D415 RST305
    |z_true - z_pred| > 1 as catastrophic outlier, then calculate
    fraction by outlier / total.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state("num_outlier_cata", default=torch.zeros(1), dist_reduce_fx="sum")
        self.add_state("total", default=torch.zeros(1), dist_reduce_fx="sum")

    def update(self, true_cat, est_cat, matching):
        for i in range(true_cat.batch_size):
            tcat_matches, ecat_matches = matching[i]
            self.total += tcat_matches.sum()

            true_red = true_cat["redshifts"][i][tcat_matches]
            est_red = est_cat["redshifts"][i][ecat_matches]
            metric_outlier_cata = torch.abs(true_red - est_red)
            self.num_outlier_cata += (metric_outlier_cata > 1).sum()

    def compute(self):
        outlier_fraction_cata = self.num_outlier_cata / self.total
        return {"redshifts/outlier_fraction_cata": outlier_fraction_cata.item()}


class RedshiftOutlierFractionCataBin(MetricBin):
    """set |z_true - z_pred| / (1 + z_true) > 0.15 as outlier,  # noqa: RST305 D415
    |z_true - z_pred| > 1 as catastrophic outlier, then calculate
    fraction by outlier / total.
    """

    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.add_state("num_outlier_cata", default=torch.zeros(self.n_bins), dist_reduce_fx="sum")
        self.add_state("total", default=torch.zeros(self.n_bins), dist_reduce_fx="sum")

    def update(self, true_cat, est_cat, matching):
        cutoffs = torch.tensor(self.mag_bin_cutoffs, device=self.device)
        for i in range(true_cat.batch_size):
            tcat_matches, ecat_matches = matching[i]

            true_red = true_cat["redshifts"][i][tcat_matches].to(self.device)
            est_red = est_cat["redshifts"][i][ecat_matches].to(self.device)
            true_mag = (
                true_cat.on_nmgy[i][..., self.mag_band].unsqueeze(-1)[tcat_matches].to(self.device)
            )
            bin_indices = torch.bucketize(true_mag, cutoffs)

            metric_outlier_cata = torch.abs(true_red - est_red)
            metric_outlier_cata = metric_outlier_cata > 1

            self.total += bin_indices.bincount(minlength=self.n_bins)
            for bin_idx, outlier in zip(bin_indices, metric_outlier_cata):
                self.num_outlier_cata[bin_idx] += outlier

    def compute(self):
        outlier_fraction_cata_per_bin = self.num_outlier_cata / self.total
        outlier_fraction_cata_per_bin_results = {
            f"redshifts/outlier_fraction_cata_bin_{i}": outlier_fraction_cata_per_bin[i].item()
            for i in range(len(outlier_fraction_cata_per_bin))
        }
        return {**outlier_fraction_cata_per_bin_results}


class RedshiftNormalizedMedianAbsDev(Metric):
    """NMAD = 1.4826 * Median(|(z_true - z_pred) / (1 + z_true)|  # noqa: RST305 D415
    - Median((z_true - z_pred) / (1 + z_true))).
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state("metrics", default=[], dist_reduce_fx="cat")

    def update(self, true_cat, est_cat, matching):
        for i in range(true_cat.batch_size):
            tcat_matches, ecat_matches = matching[i]

            true_red = true_cat["redshifts"][i][tcat_matches]
            est_red = est_cat["redshifts"][i][ecat_matches]
            metrics = (true_red - est_red) / (1 + true_red)
            self.metrics.append(metrics)

    def compute(self):
        bias = torch.median(dim_zero_cat(self.metrics))
        nmad_all = torch.abs(self.metrics - bias)
        nmad = 1.4826 * torch.median(nmad_all)
        return {"redshifts/nmad": nmad.item()}


class RedshiftNormalizedMedianAbsDevBin(MetricBin):
    """NMAD = 1.4826 * Median(|(z_true - z_pred) / (1 + z_true)| - Median((z_true - z_pred) / (1 + z_true)))."""  # noqa: E501 RST305  # pylint: disable=line-too-long

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state("metrics", default=[])  # store (z_true - z_pred) / (1 + z_true)
        for _ in range(self.n_bins):
            self.metrics.append(torch.tensor([]))

    def update(self, true_cat, est_cat, matching):
        cutoffs = torch.tensor(self.mag_bin_cutoffs, device=self.device)
        for i in range(true_cat.batch_size):
            tcat_matches, ecat_matches = matching[i]

            true_red = true_cat["redshifts"][i][tcat_matches].to(self.device)
            est_red = est_cat["redshifts"][i][ecat_matches].to(self.device)

            true_mag = (
                true_cat.on_nmgy[i][..., self.mag_band].unsqueeze(-1)[tcat_matches].to(self.device)
            )
            bin_indices = torch.bucketize(true_mag, cutoffs)

            metrics = (true_red - est_red) / (1 + true_red)

            for bin_idx, metric in zip(bin_indices, metrics):
                metric = metric.to(self.metrics[bin_idx].device)
                self.metrics[bin_idx] = torch.cat(  # noqa: WPS317
                    (
                        self.metrics[bin_idx],
                        metric.unsqueeze(0),
                    ),
                    dim=0,
                )

    def compute(self):
        # reduce across distributed system
        metrics = []
        for _ in range(self.n_bins):
            metrics.append(torch.tensor([], device=self.device))
        for i, metric in enumerate(self.metrics):
            metric = metric.to(self.device)
            metrics[i % self.n_bins] = torch.cat((metrics[i % self.n_bins], metric), dim=0)

        # compute
        bias_per_bin = [torch.median(metric) for metric in metrics]  # [tensor1, tensor2]
        nmad_all_per_bin = [torch.abs(metric - bias) for metric, bias in zip(metrics, bias_per_bin)]
        nmad = [1.4826 * torch.median(nmad_all) for nmad_all in nmad_all_per_bin]
        return {f"redshifts/nmad_bin_{i}": nmad[i].item() for i in range(len(nmad))}

    def reset(self):
        # Reset the state to the initial empty tensors
        self.metrics = []  # pylint: disable=attribute-defined-outside-init
        for _ in range(self.n_bins):
            self.metrics.append(torch.tensor([]))


class RedshiftBias(Metric):
    """bias = Median((z_true - z_pred) / (1 + z_true)))."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state(
            "bias", default=[], dist_reduce_fx="cat"
        )  # store (z_true - z_pred) / (1 + z_true)  # pylint: disable=line-too-long

    def update(self, true_cat, est_cat, matching):
        for i in range(true_cat.batch_size):
            tcat_matches, ecat_matches = matching[i]

            true_red = true_cat["redshifts"][i][tcat_matches].to(self.device)
            est_red = est_cat["redshifts"][i][ecat_matches].to(self.device)
            metrics = (true_red - est_red) / (1 + true_red)
            self.bias.append(metrics)

    def compute(self):
        bias = torch.median(dim_zero_cat(self.bias))
        return {"redshifts/bias": bias.item()}


class RedshiftBiasBin(MetricBin):
    """bias = Median((z_true - z_pred) / (1 + z_true)))."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state("bias", default=[])  # store (z_true - z_pred) / (1 + z_true)
        for _ in range(self.n_bins):
            self.bias.append(torch.tensor([]))

    def update(self, true_cat, est_cat, matching):
        cutoffs = torch.tensor(self.mag_bin_cutoffs, device=self.device)
        for i in range(true_cat.batch_size):
            tcat_matches, ecat_matches = matching[i]

            true_red = true_cat["redshifts"][i][tcat_matches].to(self.device)
            est_red = est_cat["redshifts"][i][ecat_matches].to(self.device)

            true_mag = (
                true_cat.on_nmgy[i][..., self.mag_band].unsqueeze(-1)[tcat_matches].to(self.device)
            )
            bin_indices = torch.bucketize(true_mag, cutoffs)

            metrics = (true_red - est_red) / (1 + true_red)

            for bin_idx, metric in zip(bin_indices, metrics):
                metric = metric.to(self.bias[bin_idx].device)
                self.bias[bin_idx] = torch.cat((self.bias[bin_idx], metric.unsqueeze(0)), dim=0)

    def compute(self):
        # reduce across distributed system
        metrics = []
        for _ in range(self.n_bins):
            metrics.append(torch.tensor([], device=self.device))
        for i, metric in enumerate(self.bias):
            metric = metric.to(self.device)
            metrics[i % self.n_bins] = torch.cat((metrics[i % self.n_bins], metric), dim=0)

        # compute
        bias_per_bin = [torch.median(metric) for metric in metrics]
        return {f"redshifts/bias_bin_{i}": bias_per_bin[i].item() for i in range(len(bias_per_bin))}

    def reset(self):
        # Reset the state to the initial empty tensors
        self.bias = []  # pylint: disable=attribute-defined-outside-init
        for _ in range(self.n_bins):
            self.bias.append(torch.tensor([]))


class RedshiftsCatalogMatcher(CatalogMatcher):
    def __init__(
        self,
        match_gating="n_sources",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.match_gating = match_gating

    def match_catalogs(self, true_cat, est_cat):
        """Get matched index used for metrics evaluation.

        Args:
            true_cat: true cat
            est_cat: est cat

        Returns:
            matched index based on self.match_gating
        """
        assert isinstance(true_cat, BaseTileCatalog) and isinstance(est_cat, BaseTileCatalog)
        assert true_cat.batch_size == est_cat.batch_size

        matching = []
        for i in range(true_cat.batch_size):
            if self.match_gating == "n_sources":
                gating = rearrange(true_cat["n_sources"][i], "ht wt -> ht wt 1 1")
            elif self.match_gating == "is_star":
                gating = true_cat.star_bools[i]
            elif self.match_gating == "is_galaxy":
                gating = true_cat.galaxy_bools[i]
            matching.append((gating, gating))

        return matching


def dim_zero_cat(x: Union[Tensor, List[Tensor]]) -> Tensor:
    """Concatenation along the zero dimension."""
    if isinstance(x, torch.Tensor):
        return x
    x = [y.unsqueeze(0) if y.numel() == 1 and y.ndim == 0 else y for y in x]
    if not x:  # empty list
        raise ValueError("No samples to concatenate")
    return torch.cat(x, dim=0)
