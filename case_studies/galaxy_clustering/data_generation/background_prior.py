import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits
from scipy.stats import gennorm

DES_DIR = Path(
    "/nfs/turbo/lsa-regier/scratch/gapatron/desdr-server.ncsa.illinois.edu/despublic/dr2_tiles/"
)
DES_SUBDIRS = os.listdir(DES_DIR)


class BackgroundPrior:
    def __init__(self, image_size=4800):
        super().__init__()
        self.width = image_size
        self.height = image_size
        self.center_offset = (self.width / 2) - 0.5
        self.bands = ["G", "R", "I", "Z"]
        self.n_bands = 4
        self.reference_band = 1
        self.ra_cen = 50.64516228577292
        self.dec_cen = -40.228830895890404
        self.pixels_per_mpc = 80
        self.mean_sources = 0.004
        self.G1_beta = 0.6
        self.G1_loc = 0
        self.G1_scale = 0.035
        self.G2_beta = 0.6
        self.G2_loc = 0
        self.G2_scale = 0.032
        self.pixel_scale = 0.263
        self.catalogs_sampled = 0
        self.catalogs_per_tile = 40
        self.sample_des_catalog()

    def sample_n_sources(self):
        """Sample number of background sources.

        Returns:
            Poisson sample for number of background sources
        """
        return np.random.poisson(self.mean_sources * self.width * self.height / 49)

    def sample_des_catalog(self):
        """Sample a random DES dataframe."""
        tile_choice = random.choice(DES_SUBDIRS)
        main_path = DES_DIR / Path(tile_choice) / Path(f"{tile_choice}_dr2_main.fits")
        flux_path = DES_DIR / Path(tile_choice) / Path(f"{tile_choice}_dr2_flux.fits")
        main_data = fits.getdata(main_path)
        main_df = pd.DataFrame(main_data)
        flux_data = fits.getdata(flux_path)
        flux_df = pd.DataFrame(flux_data)
        self.source_df = pd.merge(
            main_df, flux_df, left_on="COADD_OBJECT_ID", right_on="COADD_OBJECT_ID", how="left"
        )

    def sample_sources(self, n_sources):
        """Samples random sources from the current DES catalog.

        Args:
            n_sources: number of sources to sample

        Returns:
            a random sample of rows from the DES catalog
        """
        return self.source_df.sample(n_sources)

    def sample_source_types(self, sources):
        """Sample source type for each source, by thresholding DES estimator of source type.

        Args:
            sources: dataframe containing DES sources

        Returns:
            source_types: source types (0 for star, 1 for galaxy)
        """
        return np.array((sources["CLASS_STAR_R"] < 0.5))

    def sample_source_locs(self, n_sources):
        """Samples locations of background sources.
        Assumed to be uniformly distributed over the image

        Args:
            n_sources: number of background sources

        Returns:
            array of background source locations
        """
        x = np.random.uniform(0, self.width, n_sources)
        y = np.random.uniform(0, self.height, n_sources)
        return np.column_stack((x, y))

    def cartesian_to_gal(self, coordinates, pixel_scale=0.2):
        """Converts cartesian coordinates on the image to (Ra, Dec).

        Args:
            coordinates: cartesian coordinates of sources
            pixel_scale: pixel_scale to be used for the transformation

        Returns:
            galactic coordinates of sources
        """
        image_offset = (self.center_offset, self.center_offset)
        sky_center = (self.ra_cen, self.dec_cen)
        gal_coordinates = []
        for coord in coordinates:
            ra = (coord[0] - image_offset[0]) * pixel_scale / (60 * 60) + sky_center[0]
            dec = (coord[1] - image_offset[1]) * pixel_scale / (60 * 60) + sky_center[1]
            gal_coordinates.append((ra, dec))
        return gal_coordinates

    def sample_hlr(self, sources, source_types):
        """Samples half light radius for each source in the catalog.
        HLR taken from DES table
        HLR set to 1e-4 for stars

        Args:
            sources: Dataframe of DES sources
            source_types: source type for each source (0 for star, 1 for galaxy)

        Returns:
            samples for half light radius for each source
        """
        hlr_samples = self.pixel_scale * np.array(sources["FLUX_RADIUS_R"])
        return 1e-4 + (hlr_samples * (hlr_samples > 0))

    def sample_fluxes(self, sources):
        """Samples fluxes for all bands for each source.

        Args:
            sources: Dataframe of DES sources

        Returns:
            5-band array containing fluxes (clamped at 1 from below)
        """
        fluxes = np.array(
            sources[
                [
                    "FLUX_AUTO_G_x",
                    "FLUX_AUTO_R_x",
                    "FLUX_AUTO_I_x",
                    "FLUX_AUTO_Z_x",
                ]
            ]
        )

        return fluxes * (fluxes > 0)

    def sample_shape(self, num_elements):
        """Samples shape of sources.
        (G1, G2) are both assumed to have a generalized normal distribution
        We use rejection sampling to ensure:
            G1^2 + G2^2 < 1
            G1 < 0.8
            G2 < 0.8

        Args:
            num_elements: number of elements to sample

        Returns:
            samples for (G1, G2) for each source
        """
        g1_size_samples = gennorm.rvs(self.G1_beta, self.G1_loc, self.G1_scale, num_elements)
        g2_size_samples = gennorm.rvs(self.G2_beta, self.G2_loc, self.G2_scale, num_elements)
        for i in range(num_elements):
            flag_large = g1_size_samples[i] ** 2 + g2_size_samples[i] ** 2 >= 1
            flag_g1_large = g1_size_samples[i] >= 0.8
            flag_g2_large = g2_size_samples[i] >= 0.8
            flag_reject = flag_large or flag_g1_large or flag_g2_large
            while flag_reject:
                g1_size_samples[i] = gennorm.rvs(self.G1_beta, self.G1_loc, self.G1_scale, 1)[0]
                g2_size_samples[i] = gennorm.rvs(self.G2_beta, self.G2_loc, self.G2_scale, 1)[0]
                flag_large = g1_size_samples[i] ** 2 + g2_size_samples[i] ** 2 >= 1
                flag_g1_large = g1_size_samples[i] >= 0.8
                flag_g2_large = g2_size_samples[i] >= 0.8
                flag_reject = flag_large or flag_g1_large or flag_g2_large
        return g1_size_samples, g2_size_samples

    def make_background_catalog(
        self,
        flux_samples,
        hlr_samples,
        g1_size_samples,
        g2_size_samples,
        gal_source_locs,
        cartesian_source_locs,
        source_types,
    ):
        """Makes a background catalog from generated samples.

        Args:
            flux_samples: flux samples in all bands
            hlr_samples: samples of HLR
            g1_size_samples: samples of G1
            g2_size_samples: samples of G2
            gal_source_locs: samples of background locations in galactic coordinates
            cartesian_source_locs: samples of background locations in cartesian coordinates
            source_types: source types for each source (0 for star, 1 for galaxy)

        Returns:
            dataframe of catalog
        """
        mock_catalog = pd.DataFrame()
        mock_catalog["RA"] = np.array(gal_source_locs)[:, 0]
        mock_catalog["DEC"] = np.array(gal_source_locs)[:, 1]
        mock_catalog["X"] = np.array(cartesian_source_locs)[:, 0]
        mock_catalog["Y"] = np.array(cartesian_source_locs)[:, 1]
        mock_catalog["MEM"] = 0
        mock_catalog["FLUX_G"] = flux_samples[:, 0]
        mock_catalog["FLUX_R"] = flux_samples[:, 1]
        mock_catalog["FLUX_I"] = flux_samples[:, 2]
        mock_catalog["FLUX_Z"] = flux_samples[:, 3]
        mock_catalog["HLR"] = hlr_samples
        mock_catalog["FRACDEV"] = 0
        mock_catalog["G1"] = g1_size_samples
        mock_catalog["G2"] = g2_size_samples
        mock_catalog["Z"] = -1.0
        mock_catalog["SOURCE_TYPE"] = source_types.astype(int)
        return mock_catalog

    def sample_background(self):
        """Samples backgrounds.

        Returns:
            background_catalog: a single background catalogs for one image
        """
        n_sources = self.sample_n_sources()
        if self.catalogs_sampled == self.catalogs_per_tile:
            self.catalogs_sampled = 0
            self.sample_des_catalog()
        des_sources = self.sample_sources(n_sources)
        self.catalogs_sampled += 1
        cartesian_source_locs = self.sample_source_locs(n_sources)
        gal_source_locs = self.cartesian_to_gal(cartesian_source_locs)
        source_types = self.sample_source_types(des_sources)
        flux_samples = self.sample_fluxes(des_sources)
        g1_size_samples, g2_size_samples = self.sample_shape(n_sources)
        hlr_samples = self.sample_hlr(des_sources, source_types)
        return self.make_background_catalog(
            flux_samples,
            hlr_samples,
            g1_size_samples,
            g2_size_samples,
            gal_source_locs,
            cartesian_source_locs,
            source_types,
        )
