from torch.distributions import Gamma, Uniform
import torch
import pandas as pd
import numpy as np
from bliss.simulator.prior import CatalogPrior
from case_studies.redshift_estimation.catalog import RedshiftTileCatalog


class RedshiftUniformPrior(CatalogPrior):
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


class RedshiftGammaPrior(CatalogPrior):
    """Prior distribution of objects in an astronomical image.

    Inherits from CatalogPrior, adding redshift. Temporarily,
    redshift is drawn from a tight uniform distribution Unif(0.99,1.01)
    for validation.
    """

    def __init__(
        self,
        *args,
        redshift_alpha: float = 0.0,
        redshift_beta: float = 1.0,
        **kwargs,
    ):
        """Initializes CatalogPrior."""
        super().__init__(*args, **kwargs)
        self.redshift_alpha = redshift_alpha
        self.redshift_beta = redshift_beta

    def _sample_redshifts(self):
        latent_dims = (self.batch_size, self.n_tiles_h, self.n_tiles_w, self.max_sources, 1)
        return Gamma(self.redshift_alpha, self.redshift_beta).sample(latent_dims)

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


class RedshiftCSVPrior(CatalogPrior):
    """Prior distribution of objects in an astronomical image.

    Inherits from CatalogPrior, adding redshift. Load a csvfile for flux and redshift.
    """

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        """Initializes CatalogPrior."""
        super().__init__(*args, **kwargs)
        self.array = pd.read_csv('/home/../data/scratch/songju/STAR.csv').to_numpy()[:,1:7]
        latent_dims = (self.batch_size, self.n_tiles_h, self.n_tiles_w, self.max_sources, 1)
        self.__sample = np.random.choice(range(len(self.array)), size=latent_dims, replace=True)

    def _sample_redshifts(self):
        
        return torch.from_numpy(np.take(self.array[:,-1], self.__sample))

    # Override the source_type to be a star
    def _sample_source_type(self):
        latent_dims = (self.batch_size, self.n_tiles_h, self.n_tiles_w, self.max_sources, 1)
        return torch.zeros(*latent_dims)  
    
    # Overirde the star fluxes to sample from the real data
    def _sample_star_fluxes(self):
        total_flux = np.concatenate((np.take(self.array[:,0], self.__sample),np.take(self.array[:,1], self.__sample),np.take(self.array[:,2], self.__sample),np.take(self.array[:,3], self.__sample),np.take(self.array[:,4], self.__sample)),axis=-1)

        # select specified bands
        # bands = np.array(range(self.n_bands))
        # return total_flux[..., bands]
        return torch.from_numpy(total_flux)

    def sample(self) -> RedshiftTileCatalog:
        """Overrides this method from CatalogPrior to include redshift samples from prior.

        Returns:
            A dictionary of tensors. Each tensor is a particular per-tile quantity; i.e.
            the first three dimensions of each tensor are
            `(batch_size, self.n_tiles_h, self.n_tiles_w)`.
            The remaining dimensions are variable-specific.
        """
        locs = self._sample_locs()
        # Assume all sources are stars for now, cant comment this because their values are needed elsewhere even we are not using them
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
