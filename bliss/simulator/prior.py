import numpy as np
import pytorch_lightning as pl
import torch
from torch import Tensor
from torch.distributions import Gamma, Poisson, Uniform

from bliss.catalog import TileCatalog, get_is_on_from_n_sources


class ImagePrior(pl.LightningModule):
    """Prior distribution of objects in an astronomical image.

    After the module is initialized, sampling is done with the sample_prior method.
    The input parameters correspond to the number of sources, the fluxes, whether an
    object is a galaxy or star, and the distributions of galaxy and star shapes.
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
        prob_galaxy: float,
        star_flux_min: float,
        star_flux_max: float,
        star_flux_alpha: float,
        galaxy_flux_min: float,
        galaxy_flux_max: float,
        galaxy_alpha: float,
        galaxy_a_concentration: float,
        galaxy_a_loc: float,
        galaxy_a_scale: float,
        galaxy_a_bd_ratio: float,
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
            prob_galaxy: Prior probability a source is a galaxy
            star_flux_min: Prior parameter on fluxes
            star_flux_max: Prior parameter on fluxes
            star_flux_alpha: Prior parameter on fluxes (pareto parameter)
            galaxy_flux_min: Minimum flux of a galaxy
            galaxy_flux_max: Maximum flux of a galaxy
            galaxy_alpha: ?
            galaxy_a_concentration: ?
            galaxy_a_loc: ?
            galaxy_a_scale: galaxy scale
            galaxy_a_bd_ratio: galaxy bulge-to-disk ratio
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

        self.prob_galaxy = prob_galaxy

        self.star_flux_min = star_flux_min
        self.star_flux_max = star_flux_max
        self.star_flux_alpha = star_flux_alpha

        self.galaxy_flux_min = galaxy_flux_min
        self.galaxy_flux_max = galaxy_flux_max
        self.galaxy_alpha = galaxy_alpha

        self.galaxy_a_concentration = galaxy_a_concentration
        self.galaxy_a_loc = galaxy_a_loc
        self.galaxy_a_scale = galaxy_a_scale
        self.galaxy_a_bd_ratio = galaxy_a_bd_ratio

    def sample_prior(self, batch_size=0) -> TileCatalog:
        """Samples latent variables from the prior of an astronomical image.

        Args:
            batch_size: int, overrides self.batch_size if not 0.

        Returns:
            A dictionary of tensors. Each tensor is a particular per-tile quantity; i.e.
            the first three dimensions of each tensor are
            `(batch_size, self.n_tiles_h, self.n_tiles_w)`.
            The remaining dimensions are variable-specific.
        """
        # TODO: fix this hacky way of overriding batch_size
        if batch_size:
            self.batch_size = batch_size

        locs = self._sample_locs()
        galaxy_params = self._sample_galaxy_prior()
        star_fluxes = self._sample_star_fluxes()
        star_log_fluxes = star_fluxes.log()

        n_sources = self._sample_n_sources()
        galaxy_bools, star_bools = self._sample_n_galaxies_and_stars(n_sources)

        catalog_params = {
            "n_sources": n_sources,
            "galaxy_bools": galaxy_bools,
            "star_bools": star_bools,
            "locs": locs,
            "galaxy_params": galaxy_params,
            "star_fluxes": star_fluxes,
            "star_log_fluxes": star_log_fluxes,
        }

        return TileCatalog(self.tile_slen, catalog_params)

    def _sample_n_sources(self):
        latent_dims = (self.batch_size, self.n_tiles_h, self.n_tiles_w)
        n_sources = Poisson(self.mean_sources).sample(latent_dims)
        n_sources = n_sources.clamp(max=self.max_sources, min=self.min_sources)
        # long() is necessary here because the return value is used for indexing subsequently
        return n_sources.long()

    def _sample_locs(self):
        latent_dims = (self.batch_size, self.n_tiles_h, self.n_tiles_w, self.max_sources, 2)
        return Uniform(0, 1).sample(latent_dims)

    def _sample_n_galaxies_and_stars(self, n_sources):
        latent_dims = (self.batch_size, self.n_tiles_h, self.n_tiles_w, self.max_sources, 1)
        uniform_aux_var = torch.rand(*latent_dims)
        galaxy_bools = uniform_aux_var < self.prob_galaxy
        star_bools = galaxy_bools.bitwise_not()

        # gate galaxy/star booleans according n_sources
        is_on_array = get_is_on_from_n_sources(n_sources, self.max_sources)
        galaxy_bools *= is_on_array.unsqueeze(-1)
        star_bools *= is_on_array.unsqueeze(-1)

        return galaxy_bools, star_bools

    def _sample_star_fluxes(self):
        latent_dims = (self.batch_size, self.n_tiles_h, self.n_tiles_w, self.max_sources, 1)
        return self._draw_truncated_pareto(
            self.star_flux_alpha, self.star_flux_min, self.star_flux_max, latent_dims
        )

    @staticmethod
    def _draw_truncated_pareto(alpha, min_x, max_x, n_samples) -> Tensor:
        # draw pareto conditioned on being less than f_max
        u_max = 1 - (min_x / max_x) ** alpha
        uniform_samples = torch.rand(n_samples) * u_max
        return min_x / (1.0 - uniform_samples) ** (1 / alpha)

    def _sample_galaxy_prior(self):
        """Sample latent galaxy params from GalaxyPrior object."""
        latent_dims = (self.batch_size, self.n_tiles_h, self.n_tiles_w, self.max_sources, 1)

        total_flux = self._draw_truncated_pareto(
            self.galaxy_alpha, self.galaxy_flux_min, self.galaxy_flux_max, latent_dims
        )

        disk_frac = Uniform(0, 1).sample(latent_dims)
        beta_radians = Uniform(0, 2 * np.pi).sample(latent_dims)
        disk_q = Uniform(1e-8, 1).sample(latent_dims)
        bulge_q = Uniform(1e-8, 1).sample(latent_dims)

        base_dist = Gamma(self.galaxy_a_concentration, rate=1.0)
        disk_a = base_dist.sample(latent_dims) * self.galaxy_a_scale + self.galaxy_a_loc

        bulge_loc = self.galaxy_a_loc / self.galaxy_a_bd_ratio
        bulge_scale = self.galaxy_a_scale / self.galaxy_a_bd_ratio
        bulge_a = base_dist.sample(latent_dims) * bulge_scale + bulge_loc

        param_lst = [total_flux, disk_frac, beta_radians, disk_q, disk_a, bulge_q, bulge_a]
        return torch.cat(param_lst, dim=4)
