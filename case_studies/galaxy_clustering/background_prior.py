import pickle

import numpy as np
import pandas as pd
from astropy import units
from scipy.stats import gennorm

from case_studies.galaxy_clustering.utils import cluster_utils as utils


class BackgroundPrior:
    def __init__(self, image_size=4800):
        super().__init__()
        self.width = image_size
        self.height = image_size
        self.center_offset = (self.width / 2) - 0.5
        self.bands = ["G", "R", "I", "Z", "Y"]
        self.n_bands = 5
        self.reference_band = 1
        self.ra_cen = 50.64516228577292
        self.dec_cen = -40.228830895890404
        self.mass_min = (10**14.5) * (units.solMass)
        self.mass_max = (10**15.5) * (units.solMass)
        self.pixels_per_mpc = 80
        self.mean_sources = 0.004
        self.mag_ex = 1.3
        self.mag_max = 25
        self.G1_beta = 0.6
        self.G1_loc = 0
        self.G1_scale = 0.035
        self.G2_beta = 0.6
        self.G2_loc = 0
        self.G2_scale = 0.032
        with open("gal_gmm_nmgy.pkl", "rb") as f:
            self.gmm_gal = pickle.load(f)
        self.redshift_alpha = 1.65
        self.redshift_beta = 3.33
        self.redshift0 = 0.9
        self.gal_prob = 0.7
        self.light_speed = 299792.46  # in km/s
        self.sigma_DM15 = 1028.9  # in km/s

    def sample_n_sources(self):
        """Sample number of background sources.

        Returns:
            Poisson sample for number of background sources
        """
        return np.random.poisson(self.mean_sources * self.width * self.height / 49)

    def sample_source_types(self, n_sources):
        """Sample source type for each source.

        Args:
            n_sources: number of sources for each catalog

        Returns:
            source_types: source types (0 for star, 1 for galaxy)
        """
        return np.random.uniform(size=n_sources) < self.gal_prob

    def sample_background_redshift(self, num_samples):
        """Sample redshift of background galaxies.
        Assumed to be uniform in [0,4]

        Args:
            num_samples: number of samples to be generated

        Returns:
            sample for background redshift
        """
        return np.random.uniform(0, 4, num_samples)

    def sample_redshifts(self, n_sources, source_types):
        """Samples redshifts for all background sources.
        Sampled uniformly for galaxies, set to zero for stars

        Args:
            n_sources: number of background sources
            source_types: source types for each source (0 for star, 1 for galaxy)

        Returns:
            redshift_samples: list containing redshift samples for all sources
        """
        background_redshifts = self.sample_background_redshift(n_sources)
        return background_redshifts * source_types

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

    def sample_hlr(self, n_sources, source_types):
        """Samples half light radius for each source in the catalog.
        Currently assumes uniform half light radius for galaxies
        HLR set to 1e-4 for stars

        Args:
            n_sources: number of background sources
            source_types: source type for each source (0 for star, 1 for galaxy)

        Returns:
            samples for half light radius for each source
        """
        hlr_samples = np.random.uniform(0.5, 1.0, n_sources)
        hlr_samples[~source_types] = 1e-4
        return hlr_samples

    def sample_flux_r(self, redshift_samples):
        """Sample fluxes for r band.
        First samples magnitudes from an exponential distribution and subtracts from 25
        Rejection sampling to ensure magnitude is more than 15.75

        Args:
            redshift_samples: samples for redshifts of all sources

        Returns:
            flux_samples: samples for flux in r band
        """
        total_element = len(redshift_samples)
        mag_samples = self.mag_max - np.random.exponential(self.mag_ex, total_element)
        for i, _ in enumerate(mag_samples):
            while mag_samples[i] < 15.75:
                mag_samples[i] = (self.mag_max - np.random.exponential(self.mag_ex, 1))[0]
            mag_samples[i] = utils.mag_to_flux(mag_samples[i])
            mag_samples[i] *= 1 + redshift_samples[i]
        return mag_samples

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

    def sample_flux_ratios(self, gmm, num_elements):
        """Samples flux ratios from Gaussian Mixture Model (color model).

        Args:
            gmm: Gaussian Mixture Model
            num_elements: Number of elements to sample (number of sources)

        Returns:
            flux ratios for all bands
        """
        flux_logdiff, _ = gmm.sample(num_elements)
        flux_logdiff = np.clip(flux_logdiff, -2.76, 2.76)
        flux_ratio = np.exp(flux_logdiff)
        flux_prop = np.ones((flux_logdiff.shape[0], self.n_bands))
        for band in range(self.reference_band - 1, -1, -1):
            flux_prop[:, band] = flux_prop[:, band + 1] / flux_ratio[:, band]
        for band in range(self.reference_band + 1, self.n_bands):
            flux_prop[:, band] = flux_prop[:, band - 1] * flux_ratio[:, band - 1]
        return flux_prop

    def make_background_catalog(
        self,
        n_sources,
        r_flux_samples,
        hlr_samples,
        g1_size_samples,
        g2_size_samples,
        gal_source_locs,
        cartesian_source_locs,
        redshift_samples,
        source_types,
    ):
        """Makes a background catalog from generated samples.

        Args:
            n_sources: number of sources
            r_flux_samples: flux samples in R band
            hlr_samples: samples of HLR
            g1_size_samples: samples of G1
            g2_size_samples: samples of G2
            gal_source_locs: samples of background locations in galactic coordinates
            cartesian_source_locs: samples of background locations in cartesian coordinates
            redshift_samples: samples of redshifts
            source_types: source types for each source (0 for star, 1 for galaxy)

        Returns:
            dataframe of catalog
        """
        mock_catalog = pd.DataFrame()
        ratios = self.sample_flux_ratios(self.gmm_gal, n_sources)
        fluxes = np.array(r_flux_samples)[:, np.newaxis] * np.array(ratios)
        mock_catalog["RA"] = np.array(gal_source_locs)[:, 0]
        mock_catalog["DEC"] = np.array(gal_source_locs)[:, 1]
        mock_catalog["X"] = np.array(cartesian_source_locs)[:, 0]
        mock_catalog["Y"] = np.array(cartesian_source_locs)[:, 1]
        mock_catalog["MEM"] = 0
        mock_catalog["FLUX_R"] = fluxes[:, 1]
        mock_catalog["FLUX_G"] = fluxes[:, 0]
        mock_catalog["FLUX_I"] = fluxes[:, 2]
        mock_catalog["FLUX_Z"] = fluxes[:, 3]
        mock_catalog["FLUX_Y"] = fluxes[:, 4]
        mock_catalog["HLR"] = hlr_samples
        mock_catalog["FRACDEV"] = 0
        mock_catalog["G1"] = g1_size_samples
        mock_catalog["G2"] = g2_size_samples
        mock_catalog["Z"] = redshift_samples
        mock_catalog["SOURCE_TYPE"] = source_types.astype(int)
        return mock_catalog

    def sample_background(self):
        """Samples backgrounds.

        Returns:
            background_catalog: a single background catalogs for one image
        """
        n_sources = self.sample_n_sources()
        cartesian_source_locs = self.sample_source_locs(n_sources)
        gal_source_locs = self.cartesian_to_gal(cartesian_source_locs)
        source_types = self.sample_source_types(n_sources)
        redshift_samples = self.sample_redshifts(n_sources, source_types)
        r_flux_samples = self.sample_flux_r(redshift_samples)
        g1_size_samples, g2_size_samples = self.sample_shape(n_sources)
        hlr_samples = self.sample_hlr(n_sources, source_types)
        return self.make_background_catalog(
            n_sources,
            r_flux_samples,
            hlr_samples,
            g1_size_samples,
            g2_size_samples,
            gal_source_locs,
            cartesian_source_locs,
            redshift_samples,
            source_types,
        )
