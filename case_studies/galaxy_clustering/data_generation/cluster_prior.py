import pickle

import numpy as np
import pandas as pd
from astropy import units
from astropy.table import Table
from scipy.stats import gennorm

from bliss.catalog import convert_mag_to_nmgy

CLUSTER_CATALOG_PATH = "redmapper_sva1-expanded_public_v6.3_members.fits"


class ClusterPrior:
    def __init__(self, size=100, image_size=4800):
        super().__init__()
        self.size = size
        self.width = image_size
        self.height = image_size
        self.center_offset = (self.width / 2) - 0.5
        self.bands = ["G", "R", "I", "Z"]
        self.n_bands = 4
        self.reference_band = 1
        self.ra_cen = 50.64516228577292
        self.dec_cen = -40.228830895890404
        self.mass_min = (10**14.5) * (units.solMass)
        self.mass_max = (10**15.5) * (units.solMass)
        self.pixels_per_mpc = 80
        self.mean_sources = 0.004
        self.mag_ex = 1.3
        self.mag_max = 30
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
        self.light_speed = 299792.46  # in km/s
        self.sigma_DM15 = 1028.9  # in km/s
        self.star_density = 0.001

        self.full_cluster_df = Table.read(CLUSTER_CATALOG_PATH).to_pandas()
        self.cluster_indices = pd.unique(self.full_cluster_df["ID"])
        self.sample_cluster_catalog()

    def sample_cluster_catalog(self):
        """Sample a random redMaPPer catalog."""
        cluster_idx = np.random.choice(self.cluster_indices)
        self.cluster_members = self.full_cluster_df[self.full_cluster_df["ID"] == cluster_idx]

    def sample_center(self):
        """Samples cluster center on image grid.
        Sampled uniformly within a bounding box of 60% centered at image center

        Returns:
            cluster center sample
        """
        x_center = np.random.uniform(self.width * 0.2, self.width * 0.8)
        y_center = np.random.uniform(self.height * 0.2, self.height * 0.8)
        return np.array([x_center, y_center])

    def sample_cluster_locs(self, center):
        """Samples locations of clustered galaxies.
        Galaxies are assumed to be uniformly distributed in a sphere around cluster center
        We sample uniformly first in spherical coordinates following
        the sampling method outlined in https://stackoverflow.com/a/5408843
        Spherical coordinates are converted to cartesian coordinates and then one is discarded
        The radial distance is taken directly from the catalog.

        Args:
            center: cluster center in cartesian coordinates

        Returns:
            array of clustered galaxy locations
        """
        galaxy_locs_cluster = []
        center_x, center_y = center
        # convert from h-1 Mpc to pixels (assuming h = 0.7)
        radius_samples = self.pixels_per_mpc * (self.cluster_members["R"] / 0.7)
        for radius in radius_samples:
            phi = np.random.uniform(0, 2 * np.pi, 1)
            sintheta = np.random.uniform(-1, 1, 1)
            shift_x = radius * sintheta * np.cos(phi)
            shift_y = radius * sintheta * np.sin(phi)
            sampled_x = float(center_x + shift_x)
            sampled_y = float(center_y + shift_y)
            galaxy_locs_cluster.append([sampled_x, sampled_y])
        return galaxy_locs_cluster

    def cartesian_to_gal(self, coordinates, pixel_scale=0.263):
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

    def sample_hlr(self, num_elements):
        """Samples half light radius for each galaxy in the catalog.
        Currently assumes uniform half light radius

        Args:
            num_elements: number of elements to sample

        Returns:
            samples for half light radius for each galaxy in each catalog
        """
        return np.random.uniform(0.5, 1.0, num_elements)

    def sample_fluxes(self):
        """Sample fluxes from redMaPPer catalog.
        redMaPPer gives magnitudes, which are converted to picomaggies

        Returns:
            flux_samples: samples for fluxes in all bands
        """
        mags = np.array(
            self.cluster_members[
                [
                    "MAG_AUTO_G",
                    "MAG_AUTO_R",
                    "MAG_AUTO_I",
                    "MAG_AUTO_Z",
                ]
            ]
        )
        fluxes = 1000 * convert_mag_to_nmgy(mags)
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

    def make_cluster_catalog(
        self,
        flux_samples,
        hlr_samples,
        g1_size_samples,
        g2_size_samples,
        gal_locs,
        cartesian_locs,
    ):
        """Makes a single cluster catalog from generated samples.

        Args:
            flux_samples: flux samples in all bands
            hlr_samples: samples of HLR
            g1_size_samples: samples of G1
            g2_size_samples: samples of G2
            gal_locs: samples of clustered galaxy locations in galactic coordinates
            cartesian_locs: samples of clustered galaxy locations in cartesian coordinates

        Returns:
            list of dataframes (one for each catalog)
        """
        mock_catalog = pd.DataFrame()
        mock_catalog["RA"] = np.array(gal_locs)[:, 0]
        mock_catalog["DEC"] = np.array(gal_locs)[:, 1]
        mock_catalog["X"] = np.array(cartesian_locs)[:, 0]
        mock_catalog["Y"] = np.array(cartesian_locs)[:, 1]
        mock_catalog["MEM"] = 1.0
        mock_catalog["FLUX_G"] = flux_samples[:, 0]
        mock_catalog["FLUX_R"] = flux_samples[:, 1]
        mock_catalog["FLUX_I"] = flux_samples[:, 2]
        mock_catalog["FLUX_Z"] = flux_samples[:, 3]
        mock_catalog["HLR"] = hlr_samples
        mock_catalog["FRACDEV"] = 0
        mock_catalog["G1"] = g1_size_samples
        mock_catalog["G2"] = g2_size_samples
        mock_catalog["Z"] = -1.0
        mock_catalog["SOURCE_TYPE"] = 1.0
        return mock_catalog

    def sample_cluster(self):
        """Samples galaxy clusters.

        Returns:
            cluster_catalog: a single catalog containing cluster members.
        """

        self.sample_cluster_catalog()
        richness = len(self.cluster_members)
        center_sample = self.sample_center()
        cartesian_locs = self.sample_cluster_locs(center_sample)
        gal_locs = self.cartesian_to_gal(cartesian_locs)
        flux_samples = self.sample_fluxes()
        hlr_samples = self.sample_hlr(richness)
        g1_size_samples, g2_size_samples = self.sample_shape(richness)
        return self.make_cluster_catalog(
            flux_samples,
            hlr_samples,
            g1_size_samples,
            g2_size_samples,
            gal_locs,
            cartesian_locs,
        )
