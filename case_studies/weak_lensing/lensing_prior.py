import galsim
import numpy as np

try:
    import pyccl as ccl
except ModuleNotFoundError as err:
    raise ModuleNotFoundError("Please install pyccl using pip: pip install pyccl") from err
import torch
from torch.distributions import Beta, Normal, Uniform

from bliss.catalog import TileCatalog
from bliss.simulator.prior import CatalogPrior


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
        try:
            self.cosmology = ccl.cosmology.CosmologyVanillaLCDM()
        except NameError as err:
            raise NameError("Please install pyccl using pip: pip install pyccl") from err

        self.sample_method = sample_method
        self.shear_mean = shear_mean
        self.shear_std = shear_std
        self.num_knots = [num_knots, num_knots]
        self.convergence_mean = convergence_mean
        self.convergence_std = convergence_std

    def sample_shear_and_convergence_cosmology(self):
        ngrid = self.n_tiles_w
        # ngrid is tiles, convert tiles to pixels, then to arcsecs, and then degrees
        grid_size = (ngrid * self.tile_slen * self.arcsec_per_pixel) / 3600

        # redshift taken from https://github.com/LSSTDESC/CCLX/blob/master/CellsCorrelations.ipynb
        z = np.linspace(0.0, 3.0, 512)
        i_lim = 26.0  # Limiting i-band magnitude
        z0 = 0.0417 * i_lim - 0.744

        ngal = 46.0 * 100.31 * (i_lim - 25.0)  # Normalisation, galaxies/arcmin^2
        pz = 1.0 / (2.0 * z0) * (z / z0) ** 2.0 * np.exp(-z / z0)  # Redshift distribution, p(z)
        dndz = ngal * pz  # Number density distribution
        try:
            lensing_tracer = ccl.WeakLensingTracer(self.cosmology, dndz=(z, dndz))
        except NameError as err:
            raise NameError("Please install pyccl using pip: pip install pyccl") from err

        # range of values required by buildGrid
        ell = np.arange(1, 100000)

        shear_map = torch.zeros((self.batch_size, self.n_tiles_h, self.n_tiles_w, 2))
        convergence_map = torch.zeros((self.batch_size, self.n_tiles_h, self.n_tiles_w, 1))
        for i in range(self.batch_size):
            try:
                angular_cl = ccl.angular_cl(self.cosmology, lensing_tracer, lensing_tracer, ell)
            except NameError as err:
                raise NameError("Please install pyccl using pip: pip install pyccl") from err
            table = galsim.LookupTable(x=ell, f=angular_cl)
            my_ps = galsim.PowerSpectrum(table, units=galsim.degrees)
            g1, g2, kappa = my_ps.buildGrid(
                grid_spacing=grid_size / ngrid,
                ngrid=ngrid,
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
            shear, convergence = self.sample_shear_and_convergence_cosmology()

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
