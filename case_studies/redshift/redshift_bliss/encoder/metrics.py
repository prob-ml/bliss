# flake8: noqa: WPS348
from typing import List, Union

import torch
from einops import rearrange
from torch import Tensor
from torchmetrics import Metric

from bliss.catalog import BaseTileCatalog
from bliss.encoder.metrics import CatalogMatcher


def convert_nmgy_to_njymag(nmgy):
    """Convert from flux (nano-maggie) to mag (nano-jansky), which is the format used by DC2.

    For the difference between mag (Pogson magnitude) and njymag (AB magnitude), please view
    the "Flux units: maggies and nanomaggies" part of
    https://www.sdss3.org/dr8/algorithms/magnitudes.php#nmgy
    When we change the standard source to AB sources, we need to do the conversion
    described in "2.10 AB magnitudes" at
    https://pstn-001.lsst.io/fluxunits.pdf

    Args:
        nmgy: the fluxes in nanomaggies

    Returns:
        Tensor indicating fluxes in AB magnitude
    """

    return 22.5 - 2.5 * torch.log10(nmgy / 3631)


class MetricBin(Metric):
    def __init__(
        self,
        mag_band: int = 2,
        bin_cutoffs: list = None,
        bin_type: str = "njymag",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.mag_band = mag_band
        self.bin_cutoffs = bin_cutoffs
        self.n_bins = len(self.bin_cutoffs) + 1
        self.bin_type = bin_type


class RedshiftMeanSquaredError(Metric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state("sum_squared_error", default=torch.zeros(1), dist_reduce_fx="sum")
        self.add_state("total", default=torch.zeros(1), dist_reduce_fx="sum")

    def update(self, true_cat, est_cat, matching, loss):
        for i in range(true_cat.batch_size):
            tcat_matches, ecat_matches = matching[i]

            # For RedshiftsCatalogMatcher
            if isinstance(est_cat, BaseTileCatalog):
                self.total += tcat_matches.sum()
                true_red = true_cat["redshifts"][i][tcat_matches].to(self.device)
                est_red = est_cat["redshifts"][i][ecat_matches].to(self.device)
            # For CatalogMatcher
            else:
                self.total += tcat_matches.size(0)
                true_red = true_cat["redshifts"][i, tcat_matches, :].to(self.device)
                est_red = est_cat["redshifts"][i, ecat_matches, :].to(self.device)

            red_err = ((true_red - est_red).abs() ** 2).sum()

            self.sum_squared_error += red_err

    def compute(self):
        print(f"total num of pts: {self.total}")  # noqa: WPS421
        mse = self.sum_squared_error / self.total
        return {"redshifts/mse": mse.item()}


class RedshiftMeanNLL(Metric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state("sum_nll", default=torch.zeros(1), dist_reduce_fx="sum")
        self.add_state("total", default=torch.zeros(1), dist_reduce_fx="sum")

    def update(self, true_cat, est_cat, matching, loss):
        for i in range(true_cat.batch_size):
            tcat_matches, ecat_matches = matching[i]

            # For RedshiftsCatalogMatcher
            if isinstance(est_cat, BaseTileCatalog):
                self.total += tcat_matches.sum()
                nll_true = loss[i][tcat_matches].cpu().detach()

            this_nll_sum = nll_true.sum()
            self.sum_nll += this_nll_sum

    def compute(self):
        print(f"total num of pts: {self.total}")  # noqa: WPS421
        avg_nll = self.sum_nll / self.total
        return {"redshifts/nll_avg": avg_nll.item()}


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
