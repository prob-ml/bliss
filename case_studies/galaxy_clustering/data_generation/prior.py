import numpy as np
import pandas as pd
from astropy.table import Table
from scipy.stats import gennorm

from bliss.catalog import convert_mag_to_nmgy
from case_studies.galaxy_clustering.utils.cluster_utils import angular_diameter_distance

CLUSTER_CATALOG_PATH = "redmapper_sva1-expanded_public_v6.3_members.fits"
SVA_PATH = "/data/scratch/des/sva1_gold_r1.0_catalog.fits"
PHOTO_Z_PATH = "/data/scratch/des/sva1_gold_r1.0_annz2_point.fits"


class Prior:
    def __init__(self, image_size=1280):
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
        self.G1_beta = 0.6
        self.G1_loc = 0
        self.G1_scale = 0.035
        self.G2_beta = 0.6
        self.G2_loc = 0
        self.G2_scale = 0.032

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

    def make_catalog(
        self,
        flux_samples,
        hlr_samples,
        g1_size_samples,
        g2_size_samples,
        gal_locs,
        cartesian_locs,
        source_types,
        membership,
    ):
        """Makes a catalog from generated samples.

        Args:
            flux_samples: flux samples in all bands
            hlr_samples: samples of HLR
            g1_size_samples: samples of G1
            g2_size_samples: samples of G2
            gal_locs: samples of locations in galactic coordinates
            cartesian_locs: samples of locations in cartesian coordinates
            source_types: source types for each source (0 for star, 1 for galaxy)
            membership: background (0) or cluster (1)

        Returns:
            dataframe of catalog
        """
        catalog = pd.DataFrame()
        catalog["RA"] = np.array(gal_locs)[:, 0]
        catalog["DEC"] = np.array(gal_locs)[:, 1]
        catalog["X"] = np.array(cartesian_locs)[:, 0]
        catalog["Y"] = np.array(cartesian_locs)[:, 1]
        catalog["MEM"] = membership
        catalog["FLUX_G"] = flux_samples[:, 0]
        catalog["FLUX_R"] = flux_samples[:, 1]
        catalog["FLUX_I"] = flux_samples[:, 2]
        catalog["FLUX_Z"] = flux_samples[:, 3]
        catalog["HLR"] = hlr_samples
        catalog["FRACDEV"] = 0
        catalog["G1"] = g1_size_samples
        catalog["G2"] = g2_size_samples
        catalog["Z"] = -1.0
        catalog["SOURCE_TYPE"] = source_types.astype(int)
        return catalog


class ClusterPrior(Prior):
    def __init__(self, image_size=1280):
        super().__init__(image_size)

        self.full_cluster_df = Table.read(CLUSTER_CATALOG_PATH).to_pandas()
        self.cluster_indices = pd.unique(self.full_cluster_df["ID"])
        self.photo_z_catalog = Table.read(PHOTO_Z_PATH).to_pandas()
        self.sample_cluster_catalog()

    def sample_cluster_catalog(self):
        """Sample a random redMaPPer catalog."""
        cluster_idx = np.random.choice(self.cluster_indices)
        self.cluster_members = self.full_cluster_df[self.full_cluster_df["ID"] == cluster_idx]
        self.cluster_members = pd.merge(self.cluster_members, self.photo_z_catalog, how="left")

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
        radius_astro_samples = self.cluster_members["R"] / 0.7
        # redshift for most likely source that's part of the cluster
        z_mean = self.cluster_members.loc[self.cluster_members["P"].idxmax()]["Z_MEAN"]
        angular_distance = angular_diameter_distance(z_mean).value
        radius_samples = radius_astro_samples / (0.263 * angular_distance)
        radius_samples = radius_samples * (180 * 3600) / np.pi
        for radius in radius_samples:
            phi = np.random.uniform(0, 2 * np.pi, 1)
            sintheta = np.random.uniform(-1, 1, 1)
            shift_x = radius * sintheta * np.cos(phi)
            shift_y = radius * sintheta * np.sin(phi)
            sampled_x = float(center_x + shift_x)
            sampled_y = float(center_y + shift_y)
            galaxy_locs_cluster.append([sampled_x, sampled_y])
        return galaxy_locs_cluster

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

    def sample_source_types(self, richness):
        return np.ones(richness)

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
        source_types = self.sample_source_types(richness)
        g1_size_samples, g2_size_samples = self.sample_shape(richness)
        return self.make_catalog(
            flux_samples,
            hlr_samples,
            g1_size_samples,
            g2_size_samples,
            gal_locs,
            cartesian_locs,
            source_types,
            membership=1.0,
        )


class BackgroundPrior(Prior):
    def __init__(self, image_size=1280):
        super().__init__(image_size)

        self.pixel_scale = 0.263
        self.mean_sources = 0.004
        self.source_df = Table.read(SVA_PATH).to_pandas()

    def sample_n_sources(self):
        """Sample number of background sources.

        Returns:
            Poisson sample for number of background sources
        """
        return np.random.poisson(self.mean_sources * self.width * self.height / 49)

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

    def sample_hlr(self, sources):
        """Samples half light radius for each source in the catalog.
        HLR taken from DES table
        HLR set to 1e-4 for stars

        Args:
            sources: Dataframe of DES sources

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
        mags = np.array(
            sources[
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

    def sample_background(self):
        """Samples backgrounds.

        Returns:
            background_catalog: a single background catalogs for one image
        """
        n_sources = self.sample_n_sources()
        sva_sources = self.sample_sources(n_sources)
        cartesian_source_locs = self.sample_source_locs(n_sources)
        gal_source_locs = self.cartesian_to_gal(cartesian_source_locs)
        source_types = self.sample_source_types(sva_sources)
        flux_samples = self.sample_fluxes(sva_sources)
        g1_size_samples, g2_size_samples = self.sample_shape(n_sources)
        hlr_samples = self.sample_hlr(sva_sources)
        return self.make_catalog(
            flux_samples,
            hlr_samples,
            g1_size_samples,
            g2_size_samples,
            gal_source_locs,
            cartesian_source_locs,
            source_types,
            membership=0,
        )
