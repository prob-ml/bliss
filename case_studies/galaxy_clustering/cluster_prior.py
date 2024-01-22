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

from bliss.catalog import SourceType, TileCatalog, FullCatalog


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
        # Sample # of sources. Uses poisson distribution based on means sources for each tile. 
        latent_dims = (self.batch_size, 1)
        n_sources = Poisson(self.n_tiles_h * self.n_tiles_w *self.mean_sources).sample(latent_dims)
        n_sources = n_sources.clamp(max=self.n_tiles_h * self.n_tiles_w * self.max_sources, min=self.n_tiles_h * self.n_tiles_w *self.min_sources)
        return n_sources.long()

    def _sample_cluster_center(self):
        # Limits the center of cluster within a binding box so it's not on the edge.
        # 25%-75% grid within the image.

        latent_dims = (self.batch_size, 1)
        center_xs = Uniform(self.n_tiles_w*self.tile_slen*0.25, self.n_tiles_w*self.tile_slen*0.75).sample(latent_dims)
        center_ys = Uniform(self.n_tiles_h*self.tile_slen*0.25, self.n_tiles_h*self.tile_slen*0.75).sample(latent_dims)
        return torch.cat([center_xs, center_ys], dim = 1)

    def _sample_cluster_box(self):
        # Sample the Size of the cluster box. Limited to at most 50% side length.

        latent_dims = (self.batch_size, 1)
        cluster_box_x = Uniform(self.n_tiles_w*self.tile_slen*0.2, self.n_tiles_w*self.tile_slen*0.5).sample(latent_dims)
        cluster_box_y = Uniform(self.n_tiles_h*self.tile_slen*0.2, self.n_tiles_h*self.tile_slen*0.5).sample(latent_dims)
        return torch.cat([cluster_box_x, cluster_box_y], dim = 1)

    def _sample_locs(self, max_source):
        # Sample locs. 

        latent_dims = (self.batch_size, max_source, 2)
        return Uniform(0, self.tile_slen*self.n_tiles_h).sample(latent_dims)
    
    def _sample_source_type(self, max_source):
        # Sample source type.

        latent_dims = (self.batch_size, max_source, 1)
        uniform_aux_var = torch.rand(*latent_dims)
        uniform_aux_var = torch.where(uniform_aux_var > self.prob_galaxy, 0, 1)
        return uniform_aux_var
    
    def sample(self):
        boxes = self._sample_cluster_box()
        n_sources = self._sample_n_sources()
        centers = self._sample_cluster_center()
        cluster_galaxy_locs = torch.zeros((self.batch_size, int(self.n_tiles_h*self.n_tiles_w*0.25*self.max_sources), 2))
        for i in range(self.batch_size):
            # Sample the locs/params for the galaxies within the cluster.
            center = centers[i]
            left_point = max(0, center[0] - boxes[i][0]/2)
            upper = max(0, center[1] - boxes[i][1]/2)
            # Decide the maximum number of sources in the box.
            max_source_box = int(boxes[i][0]*boxes[i][1]/self.tile_slen/self.tile_slen*self.max_sources)
            temp = (max_source_box, 2)
            temp = Uniform(0,1).sample(temp)
            index = 0
            for j in range(max_source_box):
                # Randomly discard about 80% in the cluster galaxy, also make sure it doesn't outgrows the overall number (not very likely?).
                # Don't set too low. Otherwise may take over all sources and look too crowded! 
                if random.random() > 0.80 and index < n_sources[i][0]:
                    cluster_galaxy_locs[i][index][0] = left_point + boxes[i][0]*temp[j][0]
                    cluster_galaxy_locs[i][index][1] = upper + boxes[i][1]*temp[j][1]
                    index += 1
            self.n_batch_cluster_galaxy[i] = index
            # After this, we should have all locs for the galaxy within the cluster.

        cluster_galaxy_locs = cluster_galaxy_locs[:,:max(self.n_batch_cluster_galaxy),:]

        # Setting all of their types to galxies.
        cluster_types = torch.zeros((self.batch_size, max(self.n_batch_cluster_galaxy), 1))
        # Generate their fluxes.
        cluster_galaxy_fluxes, cluster_galaxy_params = self._sample_galaxy_prior(max(self.n_batch_cluster_galaxy))

        # Max number of sources across all batches that may still left for other sources
        
        max_left_n_sources = 0
        for i in range(self.batch_size):
            max_left_n_sources = max(int(max(n_sources)) - self.n_batch_cluster_galaxy[i], max_left_n_sources)
        
        # This gives the location for other sources.
        other_locs = self._sample_locs(max_left_n_sources)
        # This one contains if they are star or galaxy.
        other_types = self._sample_source_type(max_left_n_sources)
        # Combined with n_sources and the cluster info above, we should be able to solve this problem.
        star_fluxes = self._sample_star_fluxes(int(max(n_sources)))
        galaxy_fluxes, galaxy_params = self._sample_galaxy_prior(max_left_n_sources)

        final_loc = []
        final_type = []
        final_galaxy_fluxes = []
        final_galaxy_params = []

        # Concatenate them together. Could use vectorization.
        for i in range(self.batch_size):
            batch_locs = torch.cat([cluster_galaxy_locs[i,:self.n_batch_cluster_galaxy[i],:], other_locs[i, :, :]], dim =0)[:int(max(n_sources)), :]
            batch_types = torch.cat([cluster_types[i,:self.n_batch_cluster_galaxy[i],:], other_types[i, :, :]], dim =0)[:int(max(n_sources)), :]
            batch_galaxy_fluxes =  torch.cat([cluster_galaxy_fluxes[i,:self.n_batch_cluster_galaxy[i],:], galaxy_fluxes[i, :, :]], dim =0)[:int(max(n_sources)), :]
            batch_galaxy_params = torch.cat([cluster_galaxy_params[i,:self.n_batch_cluster_galaxy[i],:], galaxy_params[i, :, :]], dim =0)[:int(max(n_sources)), :]

            if final_loc == []:
                final_loc = batch_locs
            elif len(final_loc.shape) == 3:
                final_loc = torch.cat([final_loc, batch_locs.unsqueeze(0)], dim = 0)
            else:
                final_loc = torch.stack([final_loc, batch_locs], dim = 0)
            
            if final_type == []:
                final_type = batch_types
            elif len(final_type.shape) == 3:
                final_type = torch.cat([final_type, batch_types.unsqueeze(0)], dim = 0)
            else:
                final_type = torch.stack([final_type, batch_types], dim = 0)

            if final_galaxy_fluxes == []:
                final_galaxy_fluxes = batch_galaxy_fluxes
            elif len(final_galaxy_fluxes.shape) == 3:
                final_galaxy_fluxes = torch.cat([final_galaxy_fluxes, batch_galaxy_fluxes.unsqueeze(0)], dim = 0)
            else:
                final_galaxy_fluxes = torch.stack([final_galaxy_fluxes, batch_galaxy_fluxes], dim = 0)

            if final_galaxy_params == []:
                final_galaxy_params = batch_galaxy_params
            elif len(final_galaxy_params.shape) == 3:
                final_galaxy_params = torch.cat([final_galaxy_params, batch_galaxy_params.unsqueeze(0)], dim = 0)
            else:
                final_galaxy_params = torch.stack([final_galaxy_params, batch_galaxy_params], dim = 0)
        
        catalog_params = {
            "n_sources": n_sources.squeeze(),
            "source_type": final_type,
            "plocs": final_loc,
            "galaxy_fluxes": final_galaxy_fluxes,
            "galaxy_params": final_galaxy_params,
            "star_fluxes": star_fluxes,
        }

        return FullCatalog(self.n_tiles_h * self.tile_slen, self.n_tiles_w * self.tile_slen, catalog_params).to_tile_catalog(self.tile_slen, self.max_sources, True)



    # Methods for star/galaxy params are imported from the original implementaion.
    # Need to further modify to count for special param in the cluster.
    def _draw_truncated_pareto(self, exponent, truncation, loc, scale, n_samples) -> Tensor:
        # could use PyTorch's Pareto instead, but would have to transform to truncate
        samples = truncpareto.rvs(exponent, truncation, loc=loc, scale=scale, size=n_samples)
        return torch.from_numpy(samples)
    
    def _sample_star_fluxes(self, max_source):
            flux_prop = self._sample_flux_ratios(self.gmm_star, max_source)
            # latent_dims = (self.batch_size, self.n_tiles_h, self.n_tiles_w, self.max_sources, 1)
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

    def _sample_flux_ratios(self, gmm, max_dim) -> Tuple[Tensor, Tensor]:
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

        sample_dims = (self.batch_size, max_dim)
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
        flux_prop = self._sample_flux_ratios(self.gmm_gal, max_source)

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
        disk_frac = torch.unsqueeze(disk_frac, 2)
        beta_radians = torch.unsqueeze(beta_radians, 2)
        disk_q = torch.unsqueeze(disk_q, 2)
        disk_a = torch.unsqueeze(disk_a, 2)
        bulge_q = torch.unsqueeze(bulge_q, 2)
        bulge_a = torch.unsqueeze(bulge_a, 2)

        param_lst = [disk_frac, beta_radians, disk_q, disk_a, bulge_q, bulge_a]

        return select_flux, torch.cat(param_lst, dim=2)

