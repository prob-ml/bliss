import pickle

import cluster_utils as utils
import numpy as np
import pandas as pd
from astropy import units
from hmf import MassFunction
from scipy.stats import gennorm


class GalaxyClusterPrior:
    def __init__(self, size=100, image_size=4800):
        super().__init__()
        self.size = size
        self.width = image_size
        self.height = image_size
        self.center_offset = (self.width / 2) - 0.5
        self.bands = ["G", "R", "I", "Z"]
        self.n_bands = 4
        self.reference_band = 1  # !!! FIX !!!
        self.ra_cen = 50.64516228577292
        self.dec_cen = -40.228830895890404
        self.mass_min = (10**14.5) * (units.solMass)  # Minimum value of the range
        self.mass_max = (10**15.5) * (units.solMass)  # Maximum value of the range
        self.scale_pixels_per_au = 80
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
        self.tsize_poly = np.poly1d([-6.88890387e-11, 3.70584026e-5, 4.34623392e-2])
        self.redshift_alpha = 1.24
        self.redshift_beta = 1.01
        self.redshift0 = 0.51
        self.cluster_prob = 0.5

        self.col_names = [
            "RA",
            "DEC",
            "X",
            "Y",
            "MEM",
            "FLUX_R",
            "FLUX_G",
            "FLUX_I",
            "FLUX_Z",
            "TSIZE",
            "FRACDEV",
            "G1",
            "G2",
        ]

    def _cluster_threshold(self, r):
        return 3 / (2 * np.pi * (r**2))

    def _sample_mass(self):
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

    def z_to_zis(self, z, mass, size):  # SUBREDSHFT SAMPLING -- NEEDS TO BE FIXED !!!
        delta = (utils.hubble_parameter(z).value / 100 * mass / (10**15 * units.solMass)) ** (
            1 / 3
        ) * 1082.9
        # light speed
        c = 899377.37
        speed_mean = (c * (1 + z) ** 2 - c) / ((1 + z) ** 2 + 1)
        zi = np.random.normal(speed_mean, delta, size)
        return [max(0, np.sqrt((c + x) / (c - x)) - 1) for x in zi]

    def _sample_redshift(self):
        """Samples redshifts for the cluster.
        Sampled using the functional form present in cluster_utils

        Returns:
            returns self.size samples of redshift in the range [0,3]
        """
        redshift_bins = np.linspace(0.01, 3, 100)
        redshift_pdf = [
            utils.redshift_distribution(z, self.redshift_alpha, self.redshift_beta, self.redshift0)
            for z in redshift_bins
        ]
        redshift_pdf = redshift_pdf / np.sum(redshift_pdf)
        return np.random.choice(np.linspace(0.01, 3, 100), size=self.size, p=redshift_pdf)

    def _sample_radius(self, mass_samples, redshift_samples):  # !!!!!! NEEDS TO BE FIXED !!!!!!
        radius_samples = []
        for i in range(self.size):
            unscaled_r = utils.m200_to_r200(mass_samples[i], redshift_samples[i]).to(units.cm)
            radius_samples.append(
                (unscaled_r * self.scale_pixels_per_au / (3.086 * 10**24)).value
            )
        return radius_samples

    def _sample_n_cluster(self, mass_samples):
        """Samples number of clustered galaxies for each catalog based on cluster mass.
        Cluster probability is set to self.cluster_prob

        Args:
            mass_samples: samples for virial masses in units of solar mass

        Returns:
            samples of number of clustered galaxies for each corresponding virial mass
        """
        n_galaxy_cluster = []
        for i in range(self.size):
            if np.random.random() < self.cluster_prob:
                n_galaxy_cluster.append(utils.m200_to_n200(mass_samples[i]))
            else:
                n_galaxy_cluster.append(0)
        return n_galaxy_cluster

    def _sample_center(self):
        """Samples cluster center on image grid.
        Sampled uniformly within a bounding box of 60% centered at image center

        Returns:
            self.size samples of cluster centers
        """
        x_coords = np.random.uniform(self.width * 0.2, self.width * 0.8, self.size)
        y_coords = np.random.uniform(self.height * 0.2, self.height * 0.8, self.size)
        return np.vstack((x_coords, y_coords)).T

    def _sample_cluster_locs(self, center_samples, radius_samples, n_galaxy_cluster):
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
            while len(samples) < int(n_galaxy_cluster[i]):
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

    def _sample_n_galaxy(self):
        """Samples number of background galaxies.
        Assumes a Poisson distribution with 1 added to account for small images
        Ensures that each catalog has at least 1 background galaxy

        Returns:
            self.size samples of number of background galaxies
        """
        return 1 + np.random.poisson(self.mean_sources * self.width * self.height / 49, self.size)

    def _sample_galaxy_locs(self, n_galaxy):
        """Samples locations of background galaxies.
        Assumed to be uniformly distributed over the image

        Args:
            n_galaxy: list containing number of background galaxies for each catalog

        Returns:
            list of arrays of background galaxy locations for each catalog
        """
        galaxy_locs = []
        for i in range(self.size):
            x = np.random.uniform(0, self.width, n_galaxy[i])
            y = np.random.uniform(0, self.height, n_galaxy[i])
            galaxy_locs.append(np.column_stack((x, y)))
        return galaxy_locs

    def cartesian2geo(self, coordinates, pixel_scale=0.2):
        """Converts cartesian coordinates on the image to (Ra, Dec).
        Pixel scale assumed to be 0.2, needs to be calibrated with DES

        Args:
            coordinates: cartesian coordinates of galaxies
            pixel_scale: pixel_scale to be used for the transformation

        Returns:
            galactic coordinates of galaxies
        """
        image_offset = (self.center_offset, self.center_offset)
        sky_center = (self.ra_cen, self.dec_cen)
        geo_coordinates = []
        for coord_i in coordinates:
            temp = []
            for coord_ij in coord_i:
                ra = (coord_ij[0] - image_offset[0]) * pixel_scale / (60 * 60) + sky_center[0]
                dec = (coord_ij[1] - image_offset[1]) * pixel_scale / (60 * 60) + sky_center[1]
                temp.append((ra, dec))
            geo_coordinates.append(temp)
        return geo_coordinates

    def _sample_tsize(self, flux_samples):
        t_size_samples = []
        for i in range(self.size):
            t_size_samples.append(np.random.uniform(1.0, 4.0, len(flux_samples[i])))
        return t_size_samples

    def _sample_redshift_bg(self):
        redshift_bins = np.linspace(0.01, 7, 100)
        redshift_pdf = [
            utils.redshift_distribution(z, self.redshift_alpha, self.redshift_beta, self.redshift0)
            for z in redshift_bins
        ]
        redshift_pdf = redshift_pdf / np.sum(redshift_pdf)
        sampled_redshift = np.random.choice(np.linspace(0.01, 7, 100), p=redshift_pdf)
        offset_redshift = (np.random.random()) * 0.07
        return sampled_redshift + offset_redshift

    def _sample_flux(self, galaxy_locs, galaxy_locs_cluster, redshift_samples, mass_samples):
        flux_samples = []
        for i in range(self.size):
            total_element = len(galaxy_locs[i]) + len(galaxy_locs_cluster[i])
            mag_samples = self.mag_max - np.random.exponential(self.mag_ex, total_element)
            for j, _ in enumerate(mag_samples):
                while mag_samples[j] < 15.75:
                    mag_samples[j] = (self.mag_max - np.random.exponential(self.mag_ex, 1))[0]
                mag_samples[j] = utils.mag_to_flux(mag_samples[j])
                if j <= len(galaxy_locs_cluster[i]):
                    mag_samples[j] *= 1 + self.z_to_zis(redshift_samples[i], mass_samples[i], 1)[0]
                else:
                    mag_samples[j] = mag_samples[j] * (1 + self._sample_redshift_bg())
            flux_samples.append(mag_samples)
        return flux_samples

    def _sample_shape(self, galaxy_locs, galaxy_locs_cluster):
        """Samples shape of galaxies in each catalog.
        (G1, G2) are both assumed to have a generalized normal distribution
        We use rejection sampling to ensure:
            G1^2 + G2^2 <= 1
            G1 <= 0.8
            G2 <= 0.8

        Args:
            galaxy_locs: locations of background galaxies
            galaxy_locs_cluster: locations of clustered galaxies

        Returns:
            samples for (G1, G2) for each
        """
        g1_size_samples = []
        g2_size_samples = []
        for i in range(self.size):
            total_element = len(galaxy_locs[i]) + len(galaxy_locs_cluster[i])
            g1_size_samples.append(
                gennorm.rvs(self.G1_beta, self.G1_loc, self.G1_scale, total_element)
            )
            g2_size_samples.append(
                gennorm.rvs(self.G2_beta, self.G2_loc, self.G2_scale, total_element)
            )
            for j in range(total_element):
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

    def galaxy_flux_ratio(self, size):
        gmm_gal = self.gmm_gal
        flux_logdiff, _ = gmm_gal.sample(size)
        flux_logdiff = flux_logdiff[:][:, 1:]
        flux_logdiff = np.clip(flux_logdiff, -2.76, 2.76)
        flux_ratio = np.exp(flux_logdiff)
        flux_prop = np.ones((flux_logdiff.shape[0], self.n_bands))
        for band in range(self.reference_band - 1, -1, -1):
            flux_prop[:, band] = flux_prop[:, band + 1] / flux_ratio[:, band]
        for band in range(self.reference_band + 1, self.n_bands):
            flux_prop[:, band] = flux_prop[:, band - 1] * flux_ratio[:, band - 1]
        return flux_prop

    def make_catalog(
        self,
        flux_samples,
        t_size_samples,
        g1_size_samples,
        g2_size_samples,
        geo_galaxy,
        geo_galaxy_cluster,
        galaxy_locs,
        galaxy_locs_cluster,
    ):
        res = []
        for i, _ in enumerate(flux_samples):
            mock_catalog = pd.DataFrame()
            ratios = self.galaxy_flux_ratio(len(geo_galaxy_cluster[i]) + len(geo_galaxy[i]))
            fluxes = np.array(flux_samples[i])[:, np.newaxis] * np.array(ratios)
            if geo_galaxy_cluster[i] and geo_galaxy[i]:
                mock_catalog["RA"] = np.append(
                    np.array(geo_galaxy_cluster[i])[:, 0], np.array(geo_galaxy[i])[:, 0]
                )
                mock_catalog["DEC"] = np.append(
                    np.array(geo_galaxy_cluster[i])[:, 1], np.array(geo_galaxy[i])[:, 1]
                )
                mock_catalog["X"] = np.append(
                    np.array(galaxy_locs_cluster[i])[:, 0], np.array(galaxy_locs[i])[:, 0]
                )
                mock_catalog["Y"] = np.append(
                    np.array(galaxy_locs_cluster[i])[:, 1], np.array(galaxy_locs[i])[:, 1]
                )
                n_cg, n_bg = len(geo_galaxy_cluster[i]), len(geo_galaxy[i])
                mock_catalog["MEM"] = np.append(np.ones(n_cg), np.zeros(n_bg))
            else:
                mock_catalog["RA"] = np.array(geo_galaxy[i])[:, 0]
                mock_catalog["DEC"] = np.array(geo_galaxy[i])[:, 1]
                mock_catalog["X"] = np.array(galaxy_locs[i])[:, 0]
                mock_catalog["Y"] = np.array(galaxy_locs[i])[:, 1]
                mock_catalog["MEM"] = 0
            mock_catalog["FLUX_R"] = fluxes[:, 1]
            mock_catalog["FLUX_G"] = fluxes[:, 0]
            mock_catalog["FLUX_I"] = fluxes[:, 2]
            mock_catalog["FLUX_Z"] = fluxes[:, 3]
            mock_catalog["TSIZE"] = t_size_samples[i]
            mock_catalog["FRACDEV"] = 0
            mock_catalog["G1"] = g1_size_samples[i]
            mock_catalog["G2"] = g2_size_samples[i]
            res.append(mock_catalog)
        return res

    def sample(self):
        mass_samples = self._sample_mass()
        redshift_samples = self._sample_redshift()
        radius_samples = self._sample_radius(mass_samples, redshift_samples)
        n_galaxy_cluster = self._sample_n_cluster(mass_samples)
        center_samples = self._sample_center()
        galaxy_cluster_locs = self._sample_cluster_locs(
            center_samples, radius_samples, n_galaxy_cluster
        )
        n_galaxy = self._sample_n_galaxy()
        galaxy_locs = self._sample_galaxy_locs(n_galaxy)
        geo_galaxy = self.cartesian2geo(galaxy_locs)
        geo_galaxy_cluster = self.cartesian2geo(galaxy_cluster_locs)
        flux_samples = self._sample_flux(
            galaxy_locs, galaxy_cluster_locs, redshift_samples, mass_samples
        )
        tsize_samples = self._sample_tsize(flux_samples)
        g1_size_samples, g2_size_samples = self._sample_shape(galaxy_locs, galaxy_cluster_locs)
        return self.make_catalog(
            flux_samples,
            tsize_samples,
            g1_size_samples,
            g2_size_samples,
            geo_galaxy,
            geo_galaxy_cluster,
            galaxy_locs,
            galaxy_cluster_locs,
        )
