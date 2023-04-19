from typing import Optional

import numpy as np
import pytorch_lightning as pl
import torch
from einops import rearrange
from torch import Tensor
from torch.distributions import Poisson

from bliss.catalog import TileCatalog, get_is_on_from_n_sources


class GalaxyPrior:
    def __init__(
        self,
        min_flux: float,
        max_flux: float,
        alpha: float,
        a_concentration: float,
        a_loc: float,
        a_scale: float,
        a_bulge_disk_ratio: float,
    ) -> None:
        self.min_flux = min_flux
        self.max_flux = max_flux
        self.alpha = alpha

        self.a_concentration = a_concentration
        self.a_loc = a_loc
        self.a_scale = a_scale
        self.a_bulge_disk_ratio = a_bulge_disk_ratio

    def sample(self, total_latent, device="cpu"):
        total_flux = _draw_pareto(self.alpha, self.min_flux, self.max_flux, n_samples=total_latent)

        disk_frac = _uniform(0, 1, n_samples=total_latent)
        beta_radians = _uniform(0, 2 * np.pi, n_samples=total_latent)
        disk_q = _uniform(0, 1, n_samples=total_latent)
        bulge_q = _uniform(0, 1, n_samples=total_latent)

        disk_a = _gamma(self.a_concentration, self.a_loc, self.a_scale, n_samples=total_latent)
        bulge_a = _gamma(
            self.a_concentration,
            self.a_loc / self.a_bulge_disk_ratio,
            self.a_scale / self.a_bulge_disk_ratio,
            n_samples=total_latent,
        )

        param_lst = [total_flux, disk_frac, beta_radians, disk_q, disk_a, bulge_q, bulge_a]
        return torch.stack(param_lst, dim=1).to(device)


class ImagePrior(pl.LightningModule):
    """Prior distribution of objects in an astronomical image.

    After the module is initialized, sampling is done with the sample_prior method.
    The input parameters correspond to the number of sources, the fluxes, whether an
    object is a galaxy or star, and the distributions of galaxy and star shapes.

    Attributes:
        n_bands: Number of bands (colors) in the image
        min_sources: Minimum number of sources in a tile
        max_sources: Maximum number of sources in a tile
        mean_sources: Mean rate of sources appearing in a tile
        f_min: Prior parameter on fluxes
        f_max: Prior parameter on fluxes
        alpha: Prior parameter on fluxes
        prob_galaxy: Prior probability a source is a galaxy
    """

    def __init__(
        self,
        n_tiles_h: int,
        n_tiles_w: int,
        tile_slen: int,
        n_bands: int,
        batch_size: int,
        min_sources: int,
        max_sources: int,
        mean_sources: float,
        f_min: float,
        f_max: float,
        alpha: float,
        prob_galaxy: float,
        galaxy_prior: Optional[GalaxyPrior] = None,
    ):
        """Initializes ImagePrior.

        Args:
            n_tiles_h: Image height in tiles,
            n_tiles_w: Image width in tiles,
            tile_slen: Tile side length in pixels,
            n_bands: Number of bands (colors) in the image.
            batch_size: int,
            min_sources: Minimum number of sources in a tile
            max_sources: Maximum number of sources in a tile
            mean_sources: Mean rate of sources appearing in a tile
            f_min: Prior parameter on fluxes
            f_max: Prior parameter on fluxes
            alpha: Prior parameter on fluxes (pareto parameter)
            prob_galaxy: Prior probability a source is a galaxy
            galaxy_prior: Object from which galaxy latents are sampled
        """
        super().__init__()
        self.n_tiles_h = n_tiles_h
        self.n_tiles_w = n_tiles_w
        self.tile_slen = tile_slen
        self.n_bands = n_bands
        self.batch_size = batch_size

        self.min_sources = min_sources
        self.max_sources = max_sources
        self.mean_sources = mean_sources

        self.f_min = f_min
        self.f_max = f_max
        self.alpha = alpha

        self.prob_galaxy = float(prob_galaxy)
        self.galaxy_prior = galaxy_prior
        if self.prob_galaxy > 0.0:
            assert self.galaxy_prior is not None

    def sample_prior(self) -> TileCatalog:
        """Samples latent variables from the prior of an astronomical image.

        Returns:
            A dictionary of tensors. Each tensor is a particular per-tile quantity; i.e.
            the first three dimensions of each tensor are
            `(batch_size, self.n_tiles_h, self.n_tiles_w)`.
            The remaining dimensions are variable-specific.
        """
        n_sources = self._sample_n_sources(self.batch_size, self.n_tiles_h, self.n_tiles_w)
        is_on_array = get_is_on_from_n_sources(n_sources, self.max_sources)
        locs = self._sample_locs(is_on_array)

        galaxy_bools, star_bools = self._sample_n_galaxies_and_stars(is_on_array)
        galaxy_params = self._sample_galaxy_params(self.galaxy_prior, galaxy_bools)
        star_fluxes = self._sample_star_fluxes(star_bools)
        star_log_fluxes = self._get_log_fluxes(star_fluxes)

        catalog_params = {
            "n_sources": n_sources,
            "locs": locs,
            "galaxy_bools": galaxy_bools,
            "star_bools": star_bools,
            "galaxy_params": galaxy_params,
            "star_fluxes": star_fluxes,
            "star_log_fluxes": star_log_fluxes,
        }

        return TileCatalog(self.tile_slen, catalog_params)

    @staticmethod
    def _get_log_fluxes(fluxes):
        log_fluxes = torch.where(
            fluxes > 0, fluxes, torch.ones_like(fluxes)
        )  # prevent log(0) errors.
        return torch.log(log_fluxes)

    def _sample_n_sources(self, batch_size, n_tiles_h, n_tiles_w):
        # returns number of sources for each batch x tile
        # output dimension is batch_size x n_tiles_h x n_tiles_w

        # always poisson distributed.
        p = torch.full((1,), self.mean_sources, device=self.device, dtype=torch.float)
        m = Poisson(p)
        n_sources = m.sample([batch_size, n_tiles_h, n_tiles_w])

        # long() here is necessary because used for indexing and one_hot encoding.
        n_sources = n_sources.clamp(max=self.max_sources, min=self.min_sources)
        return rearrange(n_sources.long(), "b nth ntw 1 -> b nth ntw")

    def _sample_locs(self, is_on_array):
        # output dimension is batch_size x n_tiles_h x n_tiles_w x max_sources x 2

        # 2 = (x,y)
        batch_size, n_tiles_h, n_tiles_w, max_sources = is_on_array.shape
        shape = (batch_size, n_tiles_h, n_tiles_w, max_sources, 2)
        locs = torch.rand(*shape, device=is_on_array.device)
        locs *= is_on_array.unsqueeze(-1)

        return locs

    def _sample_n_galaxies_and_stars(self, is_on_array):
        # the counts returned (n_galaxies, n_stars) are of
        # shape (batch_size x n_tiles_h x n_tiles_w)
        # the booleans returned (galaxy_bools, star_bools) are of shape
        # (batch_size x n_tiles_h x n_tiles_w x max_sources x 1)
        # this last dimension is so it is consistent with other catalog values.
        batch_size, n_tiles_h, n_tiles_w, max_sources = is_on_array.shape
        uniform = torch.rand(
            batch_size,
            n_tiles_h,
            n_tiles_w,
            max_sources,
            1,
            device=is_on_array.device,
        )
        galaxy_bools = uniform < self.prob_galaxy
        star_bools = galaxy_bools.bitwise_not()
        galaxy_bools *= is_on_array.unsqueeze(-1)
        star_bools *= is_on_array.unsqueeze(-1)

        return galaxy_bools, star_bools

    def _sample_star_fluxes(self, star_bools: Tensor):
        """Samples star fluxes.

        Arguments:
            star_bools: Tensor indicating whether each object is a star or not.
                Has shape (batch_size x n_tiles_h x n_tiles_w x max_sources x 1)

        Returns:
            fluxes, tensor shape
            (batch_size x n_tiles_h x n_tiles_w x max_sources x n_bands)
        """
        device = star_bools.device
        batch_size, n_tiles_h, n_tiles_w, max_sources, _ = star_bools.shape
        shape = (batch_size, n_tiles_h, n_tiles_w, max_sources, 1)
        base_fluxes = self._draw_pareto_maxed(shape, device)

        if self.n_bands > 1:
            shape = (
                batch_size,
                n_tiles_h,
                n_tiles_w,
                max_sources,
                self.n_bands - 1,
            )
            colors = torch.randn(*shape, device=device)
            fluxes = 10 ** (colors / 2.5) * base_fluxes
            fluxes = torch.cat((base_fluxes, fluxes), dim=-1)
            fluxes *= star_bools.float()
        else:
            fluxes = base_fluxes * star_bools.float()

        return fluxes

    def _draw_pareto_maxed(self, shape, device):
        # draw pareto conditioned on being less than f_max

        u_max = self._pareto_cdf(self.f_max)
        uniform_samples = torch.rand(*shape, device=device) * u_max
        return self.f_min / (1.0 - uniform_samples) ** (1 / self.alpha)

    def _pareto_cdf(self, x):
        return 1 - (self.f_min / x) ** self.alpha

    def _sample_galaxy_params(self, galaxy_prior, galaxy_bools):
        """Sample latent galaxy params from GalaxyPrior object."""
        batch_size, n_tiles_h, n_tiles_w, max_sources, _ = galaxy_bools.shape
        total_latent = batch_size * n_tiles_h * n_tiles_w * max_sources
        if self.prob_galaxy > 0.0:
            samples = galaxy_prior.sample(total_latent, galaxy_bools.device)
        else:
            samples = torch.zeros((total_latent, 1), device=galaxy_bools.device)
        return rearrange(
            samples,
            "(b nth ntw s) g -> b nth ntw s g",
            b=batch_size,
            nth=n_tiles_h,
            ntw=n_tiles_w,
            s=max_sources,
        )


def _uniform(a, b, n_samples=1) -> Tensor:
    # uses pytorch to return a single float ~ U(a, b)
    return (a - b) * torch.rand(n_samples) + b


def _draw_pareto(alpha, min_x, max_x, n_samples=1) -> Tensor:
    # draw pareto conditioned on being less than f_max
    assert alpha is not None
    u_max = 1 - (min_x / max_x) ** alpha
    uniform_samples = torch.rand(n_samples) * u_max
    return min_x / (1.0 - uniform_samples) ** (1 / alpha)


def _gamma(concentration, loc, scale, n_samples=1):
    x = torch.distributions.Gamma(concentration, rate=1.0).sample((n_samples,))
    return x * scale + loc
