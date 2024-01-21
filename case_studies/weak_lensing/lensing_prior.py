from torch.distributions import Beta, Uniform

from bliss.simulator.prior import CatalogPrior
from case_studies.weak_lensing.lensing_catalog import LensingTileCatalog


class LensingPrior(CatalogPrior):
    def __init__(
        self,
        *args,
        shear_min: float = -0.5,
        shear_max: float = 0.5,
        convergence_a: float = 1,
        convergence_b: float = 100,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.shear_min = shear_min
        self.shear_max = shear_max
        self.convergence_a = convergence_a
        self.convergence_b = convergence_b

    def _sample_shear(self):
        latent_dims = (self.batch_size, self.n_tiles_h, self.n_tiles_w, self.max_sources, 2)
        return Uniform(self.shear_min, self.shear_max).sample(latent_dims)

    def _sample_convergence(self):
        latent_dims = (self.batch_size, self.n_tiles_h, self.n_tiles_w, self.max_sources, 1)
        return Beta(self.convergence_a, self.convergence_b).sample(latent_dims)

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
