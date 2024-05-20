import os

import galsim
import numpy as np
import torch
from torch.distributions import Beta, Normal, Uniform

from bliss.catalog import TileCatalog
from bliss.simulator.prior import CatalogPrior
from case_studies.weak_lensing import generate_angular_cl


class LensingPrior(CatalogPrior):
    def __init__(
        self,
        *args,
        arcsec_per_pixel: float,
        sample_method: str,
        shear_mean: float,
        shear_std: float,
        convergence_mean: float,
        convergence_std: float,
        num_knots: int,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        # validate that tiles height and width is the same, used as ngrid later
        self.arcsec_per_pixel = arcsec_per_pixel

        self.sample_method = sample_method
        self.shear_mean = shear_mean
        self.shear_std = shear_std
        self.num_knots = [num_knots, num_knots]
        self.convergence_mean = convergence_mean
        self.convergence_std = convergence_std

        if self.sample_method == "cosmology":
            self.grid_size = (self.n_tiles_w * self.tile_slen * self.arcsec_per_pixel) / 3600

            if os.path.exists("angular_cl.npy"):
                angular_cl = np.load("angular_cl.npy")
            else:
                generate_angular_cl.main()
                angular_cl = np.load("angular_cl.npy")

            angular_cl_table = galsim.LookupTable(x=angular_cl[0], f=angular_cl[1])
            self.power_spectrum = galsim.PowerSpectrum(angular_cl_table, units=galsim.degrees)

    def _sample_shear_and_convergence(self):
        shear_map = torch.zeros((self.batch_size, self.n_tiles_h, self.n_tiles_w, 2))
        convergence_map = torch.zeros((self.batch_size, self.n_tiles_h, self.n_tiles_w, 1))

        for i in range(self.batch_size):
            g1, g2, kappa = self.power_spectrum.buildGrid(
                grid_spacing=self.grid_size / self.n_tiles_w,
                ngrid=self.n_tiles_w,
                get_convergence=True,
                units=galsim.degrees,
            )
            gamma1 = g1 * (1 - kappa)
            gamma2 = g2 * (1 - kappa)

            shear_map[i, :, :, 0] = torch.from_numpy(gamma1)
            shear_map[i, :, :, 1] = torch.from_numpy(gamma2)
            convergence_map[i, :, :, 0] = torch.from_numpy(kappa)

        return (
            shear_map.unsqueeze(3).expand(-1, -1, -1, self.max_sources, -1),
            convergence_map.unsqueeze(3).expand(-1, -1, -1, self.max_sources, -1),
        )

    def _sample_shear(self):
        latent_dims = (self.batch_size, self.n_tiles_h, self.n_tiles_w, 2)
        if self.sample_method == "interpolate":
            # number of knots in each dimension
            corners = (self.batch_size, self.num_knots[0], self.num_knots[1], 2)

            shear_maps = Normal(self.shear_mean, self.shear_std).sample(corners)
            # want to change from 32 x 20 x 20 x 2 to 32 x 2 x 20 x 20
            shear_maps = shear_maps.reshape(
                (self.batch_size, 2, self.num_knots[0], self.num_knots[1])
            )

            shear_maps = torch.nn.functional.interpolate(
                shear_maps,
                scale_factor=(
                    self.n_tiles_h // self.num_knots[0],
                    self.n_tiles_w // self.num_knots[1],
                ),
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
        latent_dims = (self.batch_size, self.n_tiles_h, self.n_tiles_w, 1)
        if self.sample_method == "interpolate":
            # number of knots in each dimension
            corners = (self.batch_size, self.num_knots[0], self.num_knots[1], 1)
            convergence_map = Normal(self.convergence_mean, self.convergence_std).sample(corners)
            # want to change from 32 x 20 x 20 x 2 to 32 x 2 x 20 x 20
            convergence_map = convergence_map.reshape(
                (self.batch_size, 1, self.num_knots[0], self.num_knots[1])
            )

            convergence_map = torch.nn.functional.interpolate(
                convergence_map,
                scale_factor=(
                    self.n_tiles_h // self.num_knots[0],
                    self.n_tiles_w // self.num_knots[1],
                ),
                mode="bilinear",
                align_corners=True,
            )

            # want to change from 32 x 1 x 20 x 20 to 32 x 20 x 20 x 1
            convergence_map = torch.swapaxes(convergence_map, 1, 3)
            convergence_map = torch.swapaxes(convergence_map, 1, 2)
        else:
            convergence_map = Beta(self.convergence_a, self.convergence_b).sample(latent_dims)

        return convergence_map.unsqueeze(3).expand(-1, -1, -1, self.max_sources, -1)

    def sample(self) -> TileCatalog:
        """Samples latent variables from the prior of an astronomical image.

        Returns:
            A dictionary of tensors. Each tensor is a particular per-tile quantity; i.e.
            the first three dimensions of each tensor are
            `(batch_size, self.n_tiles_h, self.n_tiles_w)`.
            The remaining dimensions are variable-specific.
        """

        if self.sample_method == "interpolate":
            shear = self._sample_shear()
            convergence = self._sample_convergence()
        elif self.sample_method == "cosmology":
            shear, convergence = self._sample_shear_and_convergence()

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

        return TileCatalog(self.tile_slen, catalog_params)
