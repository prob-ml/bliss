from typing import Tuple

import numpy as np
import torch
from torch import Tensor
from torch.distributions import Beta, Gamma, Uniform

from bliss.simulator.prior import CatalogPrior
from case_studies.weak_lensing.lensing_catalog import LensingTileCatalog


class LensingPrior(CatalogPrior):
    def __init__(
        self,
        *args,
        shear_min: float,
        shear_max: float,
        convergence_a: float,
        convergence_b: float,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.shear_min = shear_min
        self.shear_max = shear_max
        self.convergence_a = convergence_a
        self.convergence_b = convergence_b

    def _sample_shear(self):
        method = "interpolate"
        latent_dims = (self.batch_size, self.n_tiles_h, self.n_tiles_w, 2)
        if method == "interpolate":
            # number of knots in each dimension
            num_knots = [4, 4]
            corners = (self.batch_size, num_knots[0], num_knots[1], 2)

            shear_maps = Uniform(self.shear_min, self.shear_max).sample(corners)
            # want to change from 32 x 20 x 20 x 2 to 32 x 2 x 20 x 20
            shear_maps = shear_maps.reshape((self.batch_size, 2, num_knots[0], num_knots[1]))

            shear_maps = torch.nn.functional.interpolate(
                shear_maps,
                scale_factor=(self.n_tiles_h // num_knots[0], self.n_tiles_w // num_knots[1]),
                mode="bilinear",
                align_corners=True,
            )

            # want to change from 32 x 2 x 20 x 20 to 32 x 20 x 20 x 2
            shear_maps = torch.swapaxes(shear_maps, 1, 3)
            shear_maps = torch.swapaxes(shear_maps, 1, 2)
        else:
            shear_maps = Uniform(self.shear_min, self.shear_max).sample(latent_dims)

        return shear_maps.unsqueeze(3).expand(-1, -1, -1, self.max_sources, -1)

    def _sample_convergence(self):
        method = "interpolate"
        latent_dims = (self.batch_size, self.n_tiles_h, self.n_tiles_w, 1)
        if method == "interpolate":
            # number of knots in each dimension
            num_knots = [4, 4]
            corners = (self.batch_size, num_knots[0], num_knots[1], 1)
            convergence_map = Beta(self.convergence_a, self.convergence_b).sample(corners)
            # want to change from 32 x 20 x 20 x 2 to 32 x 2 x 20 x 20
            convergence_map = convergence_map.reshape(
                (self.batch_size, 1, num_knots[0], num_knots[1])
            )

            convergence_map = torch.nn.functional.interpolate(
                convergence_map,
                scale_factor=(self.n_tiles_h // num_knots[0], self.n_tiles_w // num_knots[1]),
                mode="bilinear",
                align_corners=True,
            )

            # want to change from 32 x 1 x 20 x 20 to 32 x 20 x 20 x 1
            convergence_map = torch.swapaxes(convergence_map, 1, 3)
            convergence_map = torch.swapaxes(convergence_map, 1, 2)
        else:
            convergence_map = Beta(self.convergence_a, self.convergence_b).sample(latent_dims)

        return convergence_map.unsqueeze(3).expand(-1, -1, -1, self.max_sources, -1)

    def _sample_galaxy_prior(self) -> Tuple[Tensor, Tensor]:
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

        latent_dims = (self.batch_size, self.n_tiles_h, self.n_tiles_w, self.max_sources, 1)

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

    def sample(self) -> LensingTileCatalog:
        """Samples latent variables from the prior of an astronomical image.

        Returns:
            A dictionary of tensors. Each tensor is a particular per-tile quantity; i.e.
            the first three dimensions of each tensor are
            `(batch_size, self.n_tiles_h, self.n_tiles_w)`.
            The remaining dimensions are variable-specific.
        """

        shear = self._sample_shear()
        convergence = self._sample_convergence()

        locs = self._sample_locs()
        galaxy_fluxes, galaxy_params = self._sample_galaxy_prior()
        star_fluxes = self._sample_star_fluxes()

        n_sources = self._sample_n_sources()
        source_type = self._sample_source_type()

        catalog_params = {
            "shear": shear,
            "convergence": convergence,
            "n_sources": n_sources,
            "source_type": source_type,
            "locs": locs,
            "galaxy_fluxes": galaxy_fluxes,
            "galaxy_params": galaxy_params,
            "star_fluxes": star_fluxes,
        }

        return LensingTileCatalog(self.tile_slen, catalog_params)
