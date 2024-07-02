import pickle

import numpy as np
import pandas as pd
from astropy import units
from hmf import MassFunction
from scipy.stats import gennorm

from case_studies.galaxy_clustering.utils import cluster_utils as utils


class ClusterPrior:
    def __init__(self, size=100, image_size=4800):
        super().__init__()
        self.size = size
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

    def sample_mass(self):
        """Samples masses in the range [10**14.5, 10**15.5] solar masses.
        Uses the HMF (Halo Mass Function) package

        Returns:
            returns self.size samples of masses in the units of solar mass
        """
        hmf = MassFunction()
        hmf.update(Mmin=14.5, Mmax=15.5)
        mass_func = hmf.dndlnm
        mass_sample = []
        delta_mass = self.mass_max - self.mass_min
        while len(mass_sample) < self.size:
            index = np.random.randint(0, len(mass_func))
            prob = (mass_func / sum(mass_func))[index]
            if np.random.random() < prob:
                sample_mass_index = (index + np.random.random()) / len(mass_func)
                mass_sample.append(self.mass_min + (sample_mass_index * delta_mass))
        return mass_sample

    def sample_cluster_redshift(self):
        """Samples redshifts for the cluster.
        Sampled using the functional form present in cluster_utils
        Parameters empirically estimated using DES and unWISE data

        Returns:
            returns self.size samples of redshift in the range [0,1.25]
        """
        redshift_bins = np.linspace(0.01, 1.25, 100)
        redshift_pdf = [
            utils.redshift_distribution(z, self.redshift_alpha, self.redshift_beta, self.redshift0)
            for z in redshift_bins
        ]
        redshift_pdf = redshift_pdf / np.sum(redshift_pdf)
        return np.random.choice(np.linspace(0.01, 1.25, 100), size=self.size, p=redshift_pdf)

    def sample_subredshift(self, z, mass, num_samples):
        """Sample redshift of clustered galaxy given cluster redshift and virial mass.
        Assumes a normal distribution around centered at the cluster redshift.

        Args:
            z: redshift
            mass: virial mass in solar masses
            num_samples: number of samples to generate

        Returns:
            sample for redshift of clustered galaxy
        """
        hubble_z = utils.hubble_parameter(z).value / 100
        sigma_v = (hubble_z * mass / (10**15 * units.solMass)) ** (1.0 / 3.0) * self.sigma_DM15
        v_cl = self.light_speed * (((1 + z) ** 2 - 1) / ((1 + z) ** 2 + 1))
        v_samples = np.random.normal(v_cl, sigma_v, num_samples)
        return [np.sqrt((self.light_speed + v) / (self.light_speed - v)) - 1 for v in v_samples]

    def sample_redshift(self, cluster_redshift_samples, mass_samples, n_galaxy_cluster):
        """Samples redshifts for all galaxies.

        Args:
            cluster_redshift_samples: samples for cluster redshifts across catalogs
            mass_samples: samples for virial mass across catalogs
            n_galaxy_cluster: samples for number of clustered galaxies across catalogs

        Returns:
            redshift_samples: list containing redshift samples for all galaxies
        """
        redshift_samples = []
        for i in range(self.size):
            cluster_subredshifts = self.sample_subredshift(
                cluster_redshift_samples[i], mass_samples[i], n_galaxy_cluster[i]
            )
            redshift_samples.append(cluster_subredshifts)
        return redshift_samples

    def sample_radius(self, mass_samples, redshift_samples):
        """Samples radius given virial mass and redshift.
        Uses conversion function from utils

        Args:
            mass_samples: virial masses of clusters for each catalog
            redshift_samples: cluster redshifts for each catalog

        Returns:
            samples for cluster radius for each catalog (in pixels)
        """
        radius_samples = []
        for i in range(self.size):
            unscaled_r = utils.m200_to_r200(mass_samples[i], redshift_samples[i])
            radius_samples.append((unscaled_r * self.pixels_per_mpc).value)
        return radius_samples

    def sample_n_galaxy_cluster(self, mass_samples):
        """Samples number of clustered galaxies for each catalog based on cluster mass.
        Cluster probability is set to self.cluster_prob

        Args:
            mass_samples: samples for virial masses in units of solar mass

        Returns:
            samples of number of clustered galaxies for each corresponding virial mass
        """
        n_galaxy_cluster = []
        for i in range(self.size):
            n_galaxy_cluster.append(int(utils.m200_to_n200(mass_samples[i]).value))
        return n_galaxy_cluster

    def sample_center(self):
        """Samples cluster center on image grid.
        Sampled uniformly within a bounding box of 60% centered at image center

        Returns:
            self.size samples of cluster centers
        """
        x_coords = np.random.uniform(self.width * 0.2, self.width * 0.8, self.size)
        y_coords = np.random.uniform(self.height * 0.2, self.height * 0.8, self.size)
        return np.vstack((x_coords, y_coords)).T

    def sample_cluster_locs(self, center_samples, radius_samples, n_galaxy_cluster):
        """Samples locations of clustered galaxies.
        Galaxies are assumed to be uniformly distributed in a sphere around cluster center
        We sample uniformly first in spherical coordinates following
        the sampling method outlined in https://stackoverflow.com/a/5408843
        Spherical coordinates are converted to cartesian coordinates and then one is discarded

        Args:
            center_samples: samples of cluster centers
            radius_samples: cluster radius for each catalog
            n_galaxy_cluster: number of clustered galaxies in each catalog

        Returns:
            list of arrays of clustered galaxy locations for each catalog
        """
        galaxy_locs_cluster = []
        for i in range(self.size):
            center_x, center_y = center_samples[i]
            samples = []
            while len(samples) < n_galaxy_cluster[i]:
                phi = np.random.uniform(0, 2 * np.pi, 1)
                sintheta = np.random.uniform(-1, 1, 1)
                u = np.random.uniform(0, 1, 1)
                r_in_sphere = radius_samples[i] * np.cbrt(u)
                shift_x = r_in_sphere * sintheta * np.cos(phi)
                shift_y = r_in_sphere * sintheta * np.sin(phi)
                sampled_x = float(center_x + shift_x)
                sampled_y = float(center_y + shift_y)
                if 0 <= sampled_x < self.width and 0 <= sampled_y < self.height:
                    samples.append([sampled_x, sampled_y])

            galaxy_locs_cluster.append(samples)
        return galaxy_locs_cluster

    def cartesian_to_gal(self, coordinates, pixel_scale=0.263):
        """Converts cartesian coordinates on the image to (Ra, Dec).

        Args:
            coordinates: cartesian coordinates of galaxies
            pixel_scale: pixel_scale to be used for the transformation

        Returns:
            galactic coordinates of galaxies
        """
        image_offset = (self.center_offset, self.center_offset)
        sky_center = (self.ra_cen, self.dec_cen)
        gal_coordinates = []
        for coord_i in coordinates:
            temp = []
            for coord_ij in coord_i:
                ra = (coord_ij[0] - image_offset[0]) * pixel_scale / (60 * 60) + sky_center[0]
                dec = (coord_ij[1] - image_offset[1]) * pixel_scale / (60 * 60) + sky_center[1]
                temp.append((ra, dec))
            gal_coordinates.append(temp)
        return gal_coordinates

    def sample_hlr(self, num_elements):
        """Samples half light radius for each galaxy in the catalog.
        Currently assumes uniform half light radius

        Args:
            num_elements: number of elements to sample

        Returns:
            samples for half light radius for each galaxy in each catalog
        """
        hlr_samples = []
        for i in range(self.size):
            hlr_samples.append(np.random.uniform(0.5, 1.0, num_elements[i]))
        return hlr_samples

    def sample_flux_r(self, redshift_samples):
        """Sample fluxes for r band.
        First samples magnitudes from an exponential distribution and subtracts from 25
        Rejection sampling to ensure magnitude is more than 15.75

        Args:
            redshift_samples: samples for redshifts of all galaxies

        Returns:
            flux_samples: samples for flux in r band
        """
        flux_samples = []
        for i in range(self.size):
            total_element = len(redshift_samples[i])
            mag_samples = self.mag_max - np.random.exponential(self.mag_ex, total_element)
            for j, _ in enumerate(mag_samples):
                while mag_samples[j] < 20:
                    mag_samples[j] = (self.mag_max - np.random.exponential(self.mag_ex, 1))[0]
                mag_samples[j] = utils.mag_to_flux(mag_samples[j])
                mag_samples[j] *= 1 + redshift_samples[i][j]
            flux_samples.append(mag_samples)
        return flux_samples

    def sample_shape(self, num_elements):
        """Samples shape of galaxies in each catalog.
        (G1, G2) are both assumed to have a generalized normal distribution
        We use rejection sampling to ensure:
            G1^2 + G2^2 < 1
            G1 < 0.8
            G2 < 0.8

        Args:
            num_elements: number of elements to sample

        Returns:
            samples for (G1, G2) for each
        """
        g1_size_samples = []
        g2_size_samples = []
        for i in range(self.size):
            g1_size_samples.append(
                gennorm.rvs(self.G1_beta, self.G1_loc, self.G1_scale, num_elements[i])
            )
            g2_size_samples.append(
                gennorm.rvs(self.G2_beta, self.G2_loc, self.G2_scale, num_elements[i])
            )
            for j in range(num_elements[i]):
                flag_large = g1_size_samples[i][j] ** 2 + g2_size_samples[i][j] ** 2 >= 1
                flag_g1_large = g1_size_samples[i][j] >= 0.8
                flag_g2_large = g2_size_samples[i][j] >= 0.8
                flag_reject = flag_large or flag_g1_large or flag_g2_large
                while flag_reject:
                    g1_size_samples[i][j] = gennorm.rvs(
                        self.G1_beta, self.G1_loc, self.G1_scale, 1
                    )[0]
                    g2_size_samples[i][j] = gennorm.rvs(
                        self.G2_beta, self.G2_loc, self.G2_scale, 1
                    )[0]
                    flag_large = g1_size_samples[i][j] ** 2 + g2_size_samples[i][j] ** 2 >= 1
                    flag_g1_large = g1_size_samples[i][j] >= 0.8
                    flag_g2_large = g2_size_samples[i][j] >= 0.8
                    flag_reject = flag_large or flag_g1_large or flag_g2_large
        return g1_size_samples, g2_size_samples

    def sample_flux_ratios(self, gmm, size):
        """Samples flux ratios from Gaussian Mixture Model (color model).

        Args:
            gmm: Gaussian Mixture Model (galaxy or star)
            size: samples to be generated (number of galaxies)

        Returns:
            flux ratios for all bands
        """
        flux_logdiff, _ = gmm.sample(size)
        flux_logdiff = np.clip(flux_logdiff, -2.76, 2.76)
        flux_ratio = np.exp(flux_logdiff)
        flux_prop = np.ones((flux_logdiff.shape[0], self.n_bands))
        for band in range(self.reference_band - 1, -1, -1):
            flux_prop[:, band] = flux_prop[:, band + 1] * flux_ratio[:, band]
        for band in range(self.reference_band + 1, self.n_bands):
            flux_prop[:, band] = flux_prop[:, band - 1] * flux_ratio[:, band - 1]
        return flux_prop

    def make_galaxy_catalog(
        self,
        n_galaxy_cluster,
        r_flux_samples,
        hlr_samples,
        g1_size_samples,
        g2_size_samples,
        gal_locs,
        cartesian_locs,
        redshift_samples,
    ):
        """Makes list of galaxy catalogs from generated samples.

        Args:
            n_galaxy_cluster: number of clustered galaxies
            r_flux_samples: flux samples in R band
            hlr_samples: samples of HLR
            g1_size_samples: samples of G1
            g2_size_samples: samples of G2
            gal_locs: samples of clustered galaxy locations in galactic coordinates
            cartesian_locs: samples of clustered galaxy locations in cartesian coordinates
            redshift_samples: samples of redshifts

        Returns:
            list of dataframes (one for each catalog)
        """
        res = []
        for i, r_flux in enumerate(r_flux_samples):
            mock_catalog = pd.DataFrame()
            ratios = self.sample_flux_ratios(self.gmm_gal, n_galaxy_cluster[i])
            fluxes = np.array(r_flux)[:, np.newaxis] * np.array(ratios)
            mock_catalog["RA"] = np.array(gal_locs[i])[:, 0]
            mock_catalog["DEC"] = np.array(gal_locs[i])[:, 1]
            mock_catalog["X"] = np.array(cartesian_locs[i])[:, 0]
            mock_catalog["Y"] = np.array(cartesian_locs[i])[:, 1]
            mock_catalog["MEM"] = 1
            mock_catalog["FLUX_R"] = fluxes[:, 1]
            mock_catalog["FLUX_G"] = fluxes[:, 0]
            mock_catalog["FLUX_I"] = fluxes[:, 2]
            mock_catalog["FLUX_Z"] = fluxes[:, 3]
            mock_catalog["FLUX_Y"] = fluxes[:, 4]
            mock_catalog["HLR"] = hlr_samples[i]
            mock_catalog["FRACDEV"] = 0
            mock_catalog["G1"] = g1_size_samples[i]
            mock_catalog["G2"] = g2_size_samples[i]
            mock_catalog["Z"] = redshift_samples[i]
            mock_catalog["SOURCE_TYPE"] = 1
            res.append(mock_catalog)
        return res

    def make_global_catalog(self, mass_samples, redshift_samples, radius_samples, n_galaxy_cluster):
        """Makes global catalog of all clusters generated.

        Args:
            mass_samples: cluster virial masses (in solar masses)
            redshift_samples: cluster redshifts
            radius_samples: cluster virial radii (in pixels, to be converted to Mpc)
            n_galaxy_cluster: number of clustered galaxies

        Returns:
            dataframe containing all the clusters generated
            (where number of clustered galaxies is positive)
            Columns include:
                ImageID: ID of file where cluster is present
                M200: virial mass in units of 10**14 solar masses
                R200: virial radii in units of Mpc
                z_cl: cluster redshift
                N200: number of clustered galaxies within R200
        """
        df = pd.DataFrame()
        mass_samples = [mass.value / 10**14 for mass in mass_samples]
        radius_samples = [radius / self.pixels_per_mpc for radius in radius_samples]
        cluster_ids = list(np.nonzero(n_galaxy_cluster)[0])
        df["ImageID"] = cluster_ids
        df["M200"] = np.take(mass_samples, cluster_ids)
        df["R200"] = np.take(radius_samples, cluster_ids)
        df["z_cl"] = np.take(redshift_samples, cluster_ids)
        df["N200"] = np.take(n_galaxy_cluster, cluster_ids)
        return df

    def sample_cluster(self):
        """Samples galaxy clusters.

        Returns:
            cluster_catalogs: list of galaxy catalogs for each image
            global_catalogs: global galaxy catalog of cluster-wide data
        """
        mass_samples = self.sample_mass()
        cluster_redshift_samples = self.sample_cluster_redshift()
        radius_samples = self.sample_radius(mass_samples, cluster_redshift_samples)
        n_galaxy_cluster = self.sample_n_galaxy_cluster(mass_samples)
        center_samples = self.sample_center()
        cartesian_locs = self.sample_cluster_locs(center_samples, radius_samples, n_galaxy_cluster)
        redshift_samples = self.sample_redshift(
            cluster_redshift_samples, mass_samples, n_galaxy_cluster
        )
        gal_locs = self.cartesian_to_gal(cartesian_locs)
        r_flux_samples = self.sample_flux_r(redshift_samples)
        hlr_samples = self.sample_hlr(n_galaxy_cluster)
        g1_size_samples, g2_size_samples = self.sample_shape(n_galaxy_cluster)
        cluster_catalogs = self.make_galaxy_catalog(
            n_galaxy_cluster,
            r_flux_samples,
            hlr_samples,
            g1_size_samples,
            g2_size_samples,
            gal_locs,
            cartesian_locs,
            redshift_samples,
        )
        global_catalog = self.make_global_catalog(
            mass_samples, cluster_redshift_samples, radius_samples, n_galaxy_cluster
        )
        return cluster_catalogs, global_catalog
