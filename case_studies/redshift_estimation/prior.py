from torch.distributions import Uniform

from bliss.simulator.prior import CatalogPrior
from case_studies.redshift_estimation.catalog import RedshiftTileCatalog


class RedshiftPrior(CatalogPrior):
    """Prior distribution of objects in an astronomical image.

    Inherits from CatalogPrior, adding redshift. Temporarily,
    redshift is drawn from a tight uniform distribution Unif(0.99,1.01)
    for validation.
    """

    def __init__(
        self,
        *args,
        redshift_min: float = 0.0,
        redshift_max: float = 1.0,
        **kwargs,
    ):
        """Initializes CatalogPrior."""
        super().__init__(*args, **kwargs)
        self.redshift_min = redshift_min
        self.redshift_max = redshift_max

    def _sample_redshifts(self):
        latent_dims = (self.batch_size, self.n_tiles_h, self.n_tiles_w, self.max_sources, 1)
        return Uniform(self.redshift_min, self.redshift_max).sample(latent_dims)

    def sample(self) -> RedshiftTileCatalog:
        """Overrides this method from CatalogPrior to include redshift samples from prior.

        Returns:
            A dictionary of tensors. Each tensor is a particular per-tile quantity; i.e.
            the first three dimensions of each tensor are
            `(batch_size, self.n_tiles_h, self.n_tiles_w)`.
            The remaining dimensions are variable-specific.
        """
        locs = self._sample_locs()
        galaxy_fluxes, galaxy_params = self._sample_galaxy_prior()
        star_fluxes = self._sample_star_fluxes()

        n_sources = self._sample_n_sources()
        source_type = self._sample_source_type()
        redshifts = self._sample_redshifts()

        catalog_params = {
            "n_sources": n_sources,
            "source_type": source_type,
            "locs": locs,
            "galaxy_fluxes": galaxy_fluxes,
            "galaxy_params": galaxy_params,
            "star_fluxes": star_fluxes,
            "redshifts": redshifts,
        }

        return RedshiftTileCatalog(self.tile_slen, catalog_params)
