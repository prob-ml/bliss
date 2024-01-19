import pickle
import warnings
import copy
import random
from typing import Tuple

import numpy as np
import pytorch_lightning as pl
import torch
from scipy.stats import truncpareto
from torch import Tensor
from torch.distributions import Gamma, Poisson, Uniform, normal

from bliss.catalog import SourceType, TileCatalog


class ClusterPrior(pl.LightningModule):

    def __init__(
            self,
            survey_bands: list,
            n_tiles_h: int,
            n_tiles_w: int,
            tile_slen: int,
            batch_size: int,
            min_sources: int,
            max_sources: int,
            mean_sources: float,
            prob_galaxy: float,
            star_flux_exponent: float,
            star_flux_truncation: float,
            star_flux_loc: float,
            star_flux_scale: float,
            galaxy_flux_exponent: float,
            galaxy_flux_truncation: float,
            galaxy_flux_loc: float,
            galaxy_flux_scale: float,
            galaxy_a_concentration: float,
            galaxy_a_loc: float,
            galaxy_a_scale: float,
            galaxy_a_bd_ratio: float,
            star_color_model_path: str,
            gal_color_model_path: str,
            reference_band: int,
    ):
        super().__init__()
        self.n_tiles_h = n_tiles_h
        self.n_tiles_w = n_tiles_w
        self.tile_slen = tile_slen
        self.n_bands = len(survey_bands)
        # NOTE: bands have to be non-empty
        self.bands = range(self.n_bands)
        self.batch_size = batch_size

        self.min_sources = min_sources
        self.max_sources = max_sources
        self.mean_sources = mean_sources

        # TODO: refactor prior to take hydra-initialized distributions as arguments
        self.prob_galaxy = prob_galaxy

        self.star_flux_truncation = star_flux_truncation
        self.star_flux_exponent = star_flux_exponent
        self.star_flux_loc = star_flux_loc
        self.star_flux_scale = star_flux_scale

        self.galaxy_flux_exponent = galaxy_flux_exponent
        self.galaxy_flux_truncation = galaxy_flux_truncation
        self.galaxy_flux_loc = galaxy_flux_loc
        self.galaxy_flux_scale = galaxy_flux_scale

        self.galaxy_a_concentration = galaxy_a_concentration
        self.galaxy_a_loc = galaxy_a_loc
        self.galaxy_a_scale = galaxy_a_scale
        self.galaxy_a_bd_ratio = galaxy_a_bd_ratio

        self.star_color_model_path = star_color_model_path
        self.gal_color_model_path = gal_color_model_path
        self.reference_band = reference_band
        self.gmm_star, self.gmm_gal = self._load_color_models()

        self.image_size = self.tile_slen**2 * self.n_tiles_h * self.n_tiles_w
        self.galaxy_cluster_prob = 0.2
        self.n_batch_cluster_galaxy = [0]*self.batch_size

    def _sample_n_sources(self):
        latent_dims = (self.batch_size, 1)
        n_sources = Poisson(self.n_tiles_h * self.n_tiles_w *self.mean_sources).sample(latent_dims)
        n_sources = n_sources.clamp(max=self.n_tiles_h * self.n_tiles_w * self.max_sources, min=self.n_tiles_h * self.n_tiles_w *self.min_sources)
        return n_sources.long()

    def _sample_cluster_center(self):
        latent_dims = (self.batch_size, 2)
        # With in 25% boxes. Only works for square now!!!!
        center_locs = Uniform(self.n_tiles_h*self.tile_slen*0.25, self.n_tiles_h*self.tile_slen*0.75).sample(latent_dims)
        return center_locs

    def _sample_cluster_box(self):
        latent_dims = (self.batch_size, 2)
        # Size of the cluster
        cluster_box = Uniform(self.n_tiles_h*self.tile_slen*0.2, self.n_tiles_h*self.tile_slen*0.5).sample(latent_dims)
        return cluster_box

    def _sample_locs(self, max_source):
        latent_dims = (self.batch_size, max_source, 2)
        return Uniform(0, 1).sample(latent_dims)
    
    def _sample_source_type(self, max_source):
        latent_dims = (self.batch_size, max_source, 1)
        uniform_aux_var = torch.rand(*latent_dims)
        galaxy_bool = uniform_aux_var < self.prob_galaxy
        star_bool = galaxy_bool.bitwise_not()
        return SourceType.STAR * star_bool + SourceType.GALAXY * galaxy_bool
    
    def sample(self):
        boxes = self._sample_cluster_box()
        n_sources = self._sample_n_sources()
        centers = self._sample_cluster_center()
        cluster_galaxy_locs = torch.zeros((self.batch_size, int(self.n_tiles_h*self.n_tiles_w*0.25*self.max_sources)), 2)
        for i in self.batch_size:
            center = centers[i]
            left_point = max(0, center[0] - boxes[i][0]/2)
            upper = max(0, center[1] - boxes[i][1]/2)
            max_source_box = int(boxes[i][0]*boxes[i][1]/self.tile_slen/self.tile_slen*self.max_sources)
            temp = (max_source_box, 2)
            temp = Uniform(0,1).sample(temp)
            index = 0
            for j in range(max_source_box):
                # Randomly discard about 30%, also make sure it doesn't outgrows the overall number (not very likely?).
                if random.random() > 0.3 and index < n_sources[i][0]:
                    cluster_galaxy_locs[i][index][0] = left_point + boxes[i][0]*temp[j][0]
                    cluster_galaxy_locs[i][index][1] = upper + boxes[i][1]*temp[j][1]
                    index += 1
            self.n_batch_cluster_galaxy[i] = index
            # After this, we should have all locs for the galaxy within the cluster.
        # Setting all of their types to galxies.
        cluster_types = torch.zeros((self.batch_size, max(self.n_batch_cluster_galaxy[i]), 1))
        # Generate their fluxes.
        cluster_galaxy_fluxes, cluster_galaxy_params = self._sample_galaxy_prior(max(self.n_batch_cluster_galaxy[i]))



        # Max number of sources across all batches that may still left for other sources
        
        max_left_n_sources = 0
        for i in range(self.batch_size):
            max_left_n_sources = max(n_sources[i][0] - self.n_batch_cluster_galaxy[i], max_left_n_sources)
        
        # This gives the location for other sources.
        other_locs = self._sample_locs(max_left_n_sources)
        # This one contains if they are star or galaxy.
        other_types = self._sample_source_type(max_left_n_sources)
        # Combined with n_sources and the cluster info above, we should be able to solve this problem.
        star_fluxes = self._sample_star_fluxes(max_left_n_sources)
        galaxy_fluxes, galaxy_params = self._sample_galaxy_prior(max_left_n_sources)


        return None





    def _draw_truncated_pareto(self, exponent, truncation, loc, scale, n_samples) -> Tensor:
        # could use PyTorch's Pareto instead, but would have to transform to truncate
        samples = truncpareto.rvs(exponent, truncation, loc=loc, scale=scale, size=n_samples)
        return torch.from_numpy(samples)
    
    def _sample_star_fluxes(self, max_source):
            flux_prop = self._sample_flux_ratios(self.gmm_star)

            latent_dims = (self.batch_size, max_source, 1)
            ref_band_flux = self._draw_truncated_pareto(
                self.star_flux_exponent,
                self.star_flux_truncation,
                self.star_flux_loc,
                self.star_flux_scale,
                latent_dims,
            )
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

    def _sample_galaxy_prior(self, max_source) -> Tuple[Tensor, Tensor]:
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
        flux_prop = self._sample_flux_ratios(self.gmm_gal)

        latent_dims = (self.batch_size, max_source, 1)

        ref_band_flux = self._draw_truncated_pareto(
            self.galaxy_flux_exponent,
            self.galaxy_flux_truncation,
            self.galaxy_flux_loc,
            self.galaxy_flux_scale,
            latent_dims,
        )

        total_flux = flux_prop * ref_band_flux

        # select fluxes from specified bands
        bands = np.array(self.bands)
        select_flux = total_flux[..., bands]

        latent_dims = latent_dims[:-1]

        disk_frac = Uniform(0, 1).sample(latent_dims)
        beta_radians = Uniform(0, np.pi).sample(latent_dims)
        disk_q = Uniform(1e-8, 1).sample(latent_dims)
        bulge_q = Uniform(1e-8, 1).sample(latent_dims)

        base_dist = Gamma(self.galaxy_a_concentration, rate=1.0)
        disk_a = base_dist.sample(latent_dims) * self.galaxy_a_scale + self.galaxy_a_loc

        bulge_loc = self.galaxy_a_loc / self.galaxy_a_bd_ratio
        bulge_scale = self.galaxy_a_scale / self.galaxy_a_bd_ratio
        bulge_a = base_dist.sample(latent_dims) * bulge_scale + bulge_loc

        disk_frac = torch.unsqueeze(disk_frac, 4)
        beta_radians = torch.unsqueeze(beta_radians, 4)
        disk_q = torch.unsqueeze(disk_q, 4)
        disk_a = torch.unsqueeze(disk_a, 4)
        bulge_q = torch.unsqueeze(bulge_q, 4)
        bulge_a = torch.unsqueeze(bulge_a, 4)

        param_lst = [disk_frac, beta_radians, disk_q, disk_a, bulge_q, bulge_a]

        return select_flux, torch.cat(param_lst, dim=4)

