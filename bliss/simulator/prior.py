import pickle
import warnings
from typing import Tuple

import numpy as np
import pytorch_lightning as pl
import torch
from scipy.stats import truncpareto
from torch import Tensor
from torch.distributions import Gamma, Poisson, Uniform

from bliss.catalog import TileCatalog


class CatalogPrior(pl.LightningModule):
    """Prior distribution of objects in an astronomical image.

    After the module is initialized, sampling is done with the sample method.
    The input parameters correspond to the number of sources, the fluxes, whether an
    object is a galaxy or star, and the distributions of galaxy and star shapes.
    """

    def __init__(
        self,
        survey_bands: list,
        n_tiles_h: int,
        n_tiles_w: int,
        batch_size: int,
        min_sources: int,
        max_sources: int,
        mean_sources: float,
        prob_galaxy: float,
        star_flux: dict,
        galaxy_flux: dict,
        galaxy_a_concentration: float,
        galaxy_a_loc: float,
        galaxy_a_scale: float,
        galaxy_a_bd_ratio: float,
        star_color_model_path: str,
        gal_color_model_path: str,
        reference_band: int,
    ):
        """Initializes CatalogPrior.

        Args:
            survey_bands: all band-pass filters available for this survey
            n_tiles_h: Image height in tiles,
            n_tiles_w: Image width in tiles,
            batch_size: int,
            min_sources: Minimum number of sources in a tile
            max_sources: Maximum number of sources in a tile
            mean_sources: Mean rate of sources appearing in a tile
            prob_galaxy: Prior probability a source is a galaxy
            star_flux: parameters of a truncated Pareto
            galaxy_flux: parameters of a truncated Pareto
            galaxy_a_concentration: ?
            galaxy_a_loc: ?
            galaxy_a_scale: galaxy scale
            galaxy_a_bd_ratio: galaxy bulge-to-disk ratio
            star_color_model_path: path specifying star color model
            gal_color_model_path: path specifying galaxy color model
            reference_band: int denoting index of reference band
        """
        super().__init__()
        self.n_tiles_h = n_tiles_h
        self.n_tiles_w = n_tiles_w
        self.n_bands = len(survey_bands)
        # NOTE: bands have to be non-empty
        self.bands = range(self.n_bands)
        self.batch_size = batch_size

        self.min_sources = min_sources
        self.max_sources = max_sources
        self.mean_sources = mean_sources

        self.prob_galaxy = prob_galaxy

        self.star_flux = star_flux
        self.galaxy_flux = galaxy_flux

        self.galaxy_a_concentration = galaxy_a_concentration
        self.galaxy_a_loc = galaxy_a_loc
        self.galaxy_a_scale = galaxy_a_scale
        self.galaxy_a_bd_ratio = galaxy_a_bd_ratio

        self.star_color_model_path = star_color_model_path
        self.gal_color_model_path = gal_color_model_path
        self.reference_band = reference_band
        self.gmm_star, self.gmm_gal = self._load_color_models()

    def sample(self) -> TileCatalog:
        """Samples latent variables from the prior of an astronomical image.

        Returns:
            A dictionary of tensors. Each tensor is a particular per-tile quantity; i.e.
            the first three dimensions of each tensor are
            `(batch_size, self.n_tiles_h, self.n_tiles_w)`.
            The remaining dimensions are variable-specific.
        """
        d = self._sample_galaxy_shape()

        d["locs"] = self._sample_locs()
        d["n_sources"] = self._sample_n_sources()
        d["source_type"] = self._sample_source_type()

        star_fluxes = self._sample_fluxes(self.gmm_star, self.star_flux)
        galaxy_fluxes = self._sample_fluxes(self.gmm_gal, self.galaxy_flux)
        d["fluxes"] = torch.where(d["source_type"], galaxy_fluxes, star_fluxes)

        return TileCatalog(d)

    def _sample_n_sources(self):
        latent_dims = (self.batch_size, self.n_tiles_h, self.n_tiles_w)
        n_sources = Poisson(self.mean_sources).sample(latent_dims)
        n_sources = n_sources.clamp(max=self.max_sources, min=self.min_sources)
        # long() is necessary here because the return value is used for indexing subsequently
        return n_sources.long()

    def _sample_locs(self):
        latent_dims = (self.batch_size, self.n_tiles_h, self.n_tiles_w, self.max_sources, 2)
        return Uniform(0, 1).sample(latent_dims)

    def _sample_source_type(self):
        latent_dims = (self.batch_size, self.n_tiles_h, self.n_tiles_w, self.max_sources, 1)
        uniform_aux_var = torch.rand(*latent_dims)
        return uniform_aux_var < self.prob_galaxy

    def _draw_truncated_pareto(self, params, n_samples) -> Tensor:
        # could use PyTorch's Pareto instead, but would have to transform to truncate
        samples = truncpareto.rvs(
            params["exponent"],
            params["truncation"],
            loc=params["loc"],
            scale=params["scale"],
            size=n_samples,
        )
        return torch.from_numpy(samples)

    def _sample_fluxes(self, gmm, flux_params):
        flux_prop = self._sample_flux_ratios(gmm)
        latent_dims = (self.batch_size, self.n_tiles_h, self.n_tiles_w, self.max_sources, 1)
        ref_band_flux = self._draw_truncated_pareto(flux_params, latent_dims)
        total_flux = ref_band_flux * flux_prop

        # select specified bands
        bands = np.array(range(self.n_bands))
        return total_flux[..., bands]

    def _load_color_models(self):
        # Load models from disk
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            with open(self.star_color_model_path, "rb") as f:
                gmm_star = pickle.load(f)
            with open(self.gal_color_model_path, "rb") as f:
                gmm_gal = pickle.load(f)
        return gmm_star, gmm_gal

    def _sample_flux_ratios(self, gmm) -> Tuple[Tensor, Tensor]:
        """Sample and compute all star, galaxy fluxes based on real image data.

        Instead of pareto-sampling fluxes for each band, we pareto-sample `b`-band flux values,
        using prior `b`-band star/galaxy flux parameters, then apply other-band-to-`b`-band flux
        ratios to get other-band flux values. This is primarily because it is not always the case
        that, e.g.,
            flux_r ~ Pareto, flux_g ~ Pareto => flux_r | flux_g ~ Pareto.
        This function is survey-specific as the 'b'-band index depends on the survey.

        Args:
            gmm: Gaussian mixture model of flux ratios (either star or galaxy)

        Returns:
            stars_fluxes (Tensor): (b x th x tw x ms x nbands) Tensor containing all star flux
                ratios for current batch
            gals_fluxes (Tensor): (b x th x tw x ms x nbands) Tensor containing all gal fluxes
                ratios for current batch
        """
        sample_dims = (self.batch_size, self.n_tiles_h, self.n_tiles_w, self.max_sources)
        flux_logdiff, _ = gmm.sample(np.prod(sample_dims))

        # A log difference of +/- 2.76 correpsonds to a 3 magnitude difference (e.g. 18 vs 21),
        # or equivalently an 15.8x flux ratio.
        # It's unlikely that objects will have a ratio larger than this.
        flux_logdiff = np.clip(flux_logdiff, -2.76, 2.76)
        flux_ratio = np.exp(flux_logdiff)

        # Computes the flux in each band as a proportion of the reference band flux
        flux_prop = torch.ones(flux_logdiff.shape[0], self.n_bands)
        for band in range(self.reference_band - 1, -1, -1):
            flux_prop[:, band] = flux_prop[:, band + 1] / flux_ratio[:, band]
        for band in range(self.reference_band + 1, self.n_bands):
            flux_prop[:, band] = flux_prop[:, band - 1] * flux_ratio[:, band - 1]

        # Reshape drawn values into appropriate form
        sample_dims = sample_dims + (self.n_bands,)
        return flux_prop.view(sample_dims)

    def _sample_galaxy_shape(self) -> Tuple[Tensor, Tensor]:
        """Sample the latent galaxy params.

        Returns:
            Tuple[Tensor]: A tuple of galaxy fluxes (per band) and galsim parameters, including.
                - disk_frac: the fraction of flux attributed to the disk (rest goes to bulge)
                - beta_radians: the angle of shear in radians
                - disk_q: the minor-to-major axis ratio of the disk
                - a_d: semi-major axis of disk
                - bulge_q: minor-to-major axis ratio of the bulge
                - a_b: semi-major axis of bulge
        """
        latent_dims = (self.batch_size, self.n_tiles_h, self.n_tiles_w, self.max_sources, 1)

        disk_frac = Uniform(0, 1).sample(latent_dims)
        beta_radians = Uniform(0, np.pi).sample(latent_dims)
        disk_q = Uniform(1e-8, 1).sample(latent_dims)
        bulge_q = Uniform(1e-8, 1).sample(latent_dims)

        base_dist = Gamma(self.galaxy_a_concentration, rate=1.0)
        disk_a = base_dist.sample(latent_dims) * self.galaxy_a_scale + self.galaxy_a_loc

        bulge_loc = self.galaxy_a_loc / self.galaxy_a_bd_ratio
        bulge_scale = self.galaxy_a_scale / self.galaxy_a_bd_ratio
        bulge_a = base_dist.sample(latent_dims) * bulge_scale + bulge_loc

        return {
            "galaxy_disk_frac": disk_frac,
            "galaxy_beta_radians": beta_radians,
            "galaxy_disk_q": disk_q,
            "galaxy_a_d": disk_a,
            "galaxy_bulge_q": bulge_q,
            "galaxy_a_b": bulge_a,
        }
