import matplotlib.pyplot as plt
from hmf import MassFunction
import numpy as np
import pandas as pd
from astropy.cosmology import WMAP7, FlatLambdaCDM
from scipy.constants import G
import pickle
import fitsio as fio
import images
from scipy.stats import lognorm, gennorm
import tutorial.synthetic.tools as tools
import tutorial.synthetic.render.frame as frame
import tutorial.synthetic.render.render as render
import os
import multiprocessing

class Cluster_Prior():
    def __init__(self):
        super().__init__()
        self.size = 100
        self.width = 5000
        self.height = 5000
        self.bands = ["G", "R", "I", "Z"]
        self.n_bands = 4
        self.reference_band = 1
        self.ra_cen = 50.64516228577292
        self.dec_cen = -40.228830895890404   
        self.mass_min = 10**14.5 * 1.989*10**33 # Minimum value of the range solar mass
        self.mass_max = 10**15.5 * 1.989*10**33 # Maximum value of the range
        self.scale_pixels_per_au = 80 
        self.mean_sources = 0.004
        self.tsize_s = 0.64
        self.tsize_loc=0.017
        self.tsize_scale=0.23
        self.mag_ex = 1.3
        self.mag_max = 25
        self.bg_min_z = 0
        self.bg_max_z = 0.3
        self.cluster_min_z = 0.2
        self.cluster_max_z = 0.5
        self.G1_beta = 0.6
        self.G1_loc = 0
        self.G1_scale = 0.035 
        self.G2_beta = 0.6
        self.G2_loc = 0
        self.G2_scale = 0.032 
        self.tiles_width = 25
        self.tiles_height = 25
        with open("gal_gmm_nmgy.pkl", "rb") as f:
            self.gmm_gal = pickle.load(f)
        self.tsize_poly = np.poly1d([-6.88890387e-11,  3.70584026e-05,  4.34623392e-02])
        self.folder_path = "data5/"
        self.threadings = 8
        self.z_pdf = [4.4230e-03, 1.4160e-02, 2.3567e-02, 3.0799e-02, 3.6851e-02,
       4.1139e-02, 4.4207e-02, 4.5863e-02, 4.6542e-02, 4.6706e-02,
       4.5852e-02, 4.5019e-02, 4.2973e-02, 4.1131e-02, 3.9549e-02,
       3.7211e-02, 3.5322e-02, 3.2596e-02, 3.0429e-02, 2.8230e-02,
       2.6276e-02, 2.4187e-02, 2.2367e-02, 2.0503e-02, 1.8803e-02,
       1.7241e-02, 1.5551e-02, 1.4083e-02, 1.2708e-02, 1.1563e-02,
       1.0635e-02, 9.4840e-03, 8.7660e-03, 7.9220e-03, 7.0570e-03,
       6.4690e-03, 5.7690e-03, 5.1020e-03, 4.5980e-03, 4.1850e-03,
       3.8310e-03, 3.2900e-03, 3.0160e-03, 2.5740e-03, 2.3900e-03,
       2.0760e-03, 1.8210e-03, 1.6950e-03, 1.5270e-03, 1.3000e-03,
       1.2230e-03, 1.0660e-03, 9.4100e-04, 8.1700e-04, 7.4800e-04,
       6.5600e-04, 6.1900e-04, 5.5200e-04, 4.5300e-04, 4.2200e-04,
       3.7600e-04, 3.2500e-04, 3.0700e-04, 2.4900e-04, 2.3200e-04,
       2.0500e-04, 1.7800e-04, 1.7200e-04, 1.2900e-04, 1.1200e-04,
       1.1000e-04, 7.8000e-05, 8.4000e-05, 8.9000e-05, 7.3000e-05,
       5.0000e-05, 4.7000e-05, 3.7000e-05, 3.1000e-05, 3.6000e-05,
       2.4000e-05, 3.1000e-05, 1.5000e-05, 2.3000e-05, 2.0000e-05,
       9.0000e-06, 1.3000e-05, 1.4000e-05, 1.4000e-05, 8.0000e-06,
       9.0000e-06, 1.2000e-05, 6.0000e-06, 5.0000e-06, 5.0000e-06,
       3.0000e-06, 2.0000e-06, 1.0000e-06, 3.0000e-06, 8.0000e-06]
        self.cluster_prob = 0.2

    def _sample_mass(self):
        hmf = MassFunction() 
        hmf.update(Mmin=14.5, Mmax = 15.5) 
        mass_func = hmf.dndlnm
        mass_sample = []
        while(len(mass_sample) < self.size):
            index = np.random.randint(0, len(mass_func))
            prob = (mass_func/sum(mass_func))[index]
            if np.random.random() < prob:
                mass_sample.append((self.mass_max-self.mass_min)/len(mass_func)*(index+np.random.random()) + self.mass_min)
        return mass_sample
    
    def Z_to_zis(self, z, mass, size):
            delta = (WMAP7.H(z).value/100*mass/(10**15*1.989*10**33))**(1/3)*1082.9
            # light speed 
            c = 899377.37
            speed_mean = (c*(1+z)**2 - c)/((1+z)**2 + 1)
            zis = np.random.normal(speed_mean, delta, size)
            zis = [max(0, np.sqrt((c+x)/(c-x)) - 1) for x in zis]
            return zis

    def Z_M_to_R(self, mass, z):
        pho_z = WMAP7.critical_density(z).value
        return (mass/(4/3*np.pi*pho_z))**(1/3)
    
    def _sample_redshift(self):
        redshift_samples = np.random.choice(np.linspace(0.01, 7, 100), size = self.size, p = self.z_pdf)
        for i in range(self.size):
            redshift_samples[i] += (np.random.random())*0.07 
        return redshift_samples
    
    def _sample_radius(self, mass_samples, redshift_samples):
        radius_samples = []
        for i in range(self.size):
            radius_samples.append(self.Z_M_to_R(mass_samples[i], redshift_samples[i])/(3.086*10**24) * self.scale_pixels_per_au)
        return radius_samples
    
    def _sample_n_cluster(self, mass_samples):
        n_galaxy_cluster = []
        for i in range(self.size):
            if np.random.random() > self.cluster_prob:
                n_galaxy_cluster.append(int(((mass_samples[i]/(1.989*10**33))/(1.4*10**13))**(1/1.35)*20))
            else:
                n_galaxy_cluster.append(0)
        return n_galaxy_cluster
    
    def _sample_center(self):
        center_sample = []
        x_coords = np.random.uniform(self.width * 0.2, self.width * 0.8, self.size)
        y_coords = np.random.uniform(self.height * 0.2, self.height * 0.8, self.size)
        center_sample = np.vstack((x_coords, y_coords)).T
        return center_sample
    
    def _sample_cluster_locs(self, center_samples, radius_samples, n_galaxy_cluster):
        galaxy_locs_cluster = []
        for i in range(self.size):
            center_x, center_y = center_samples[i]
            samples = []
            while(len(samples) < int(n_galaxy_cluster[i])):
                angles = np.random.uniform(0, 2 * np.pi, 1)
                radii = np.random.uniform(0, radius_samples[i], 1)  
                sampled_x = float(center_x + radii * np.cos(angles))  
                sampled_y = float(center_y + radii * np.sin(angles))
                if sampled_x >= 0 and sampled_x < self.width and sampled_y >= 0 and sampled_y < self.height:
                    if np.random.uniform(0, 3/(2*np.pi*(radius_samples[i])**2)) < 3/(2*np.pi*(radius_samples[i])**3) * np.sqrt(radius_samples[i]**2 - radii**2):
                        samples.append([sampled_x, sampled_y])

            print(len(galaxy_locs_cluster))
            galaxy_locs_cluster.append(samples)
        return galaxy_locs_cluster
    
    def _sample_n_galaxy(self):
        return np.random.poisson(self.mean_sources*self.width*self.height/49, self.size)
    
    def _sample_galaxy_locs(self, n_galaxy):
        galaxy_locs = []
        for i in range(self.size):
            x = np.random.uniform(0, self.width, n_galaxy[i])
            y = np.random.uniform(0, self.height, n_galaxy[i])
            galaxy_locs.append(np.column_stack((x, y)))
        return galaxy_locs
    
    def cartesian2geo(self, coordinates, pixel_scale=0.2, image_offset=(2499.5, 2499.5)):
        sky_center=(self.ra_cen, self.dec_cen)
        geo_coordinates = []
        for i in range(len(coordinates)):
            temp = []
            for j in range(len(coordinates[i])):
                ra = (coordinates[i][j][0] - image_offset[0])*pixel_scale / (60*60) + sky_center[0]
                desc = (coordinates[i][j][1] - image_offset[1])*pixel_scale / (60*60) + sky_center[1]
                temp.append((ra, desc))
            geo_coordinates.append(temp)
        return geo_coordinates
    
    def _sample_TSIZE(self, flux_samples):
        t_size_samples = []
        for i in range(self.size):
                t_size_samples.append(self.tsize_poly(flux_samples[i]))
        return t_size_samples
    
    def _sample_redshift_bg(self):
        return np.random.choice(np.linspace(0.01, 7, 100), p = self.z_pdf)+(np.random.random())*0.07 
    
    def _sample_flux(self, galaxy_locs, galaxy_locs_cluster, redshift_samples, mass_samples):
        flux_samples = []
        for i in range(self.size):
            total_element = len(galaxy_locs[i]) + len(galaxy_locs_cluster[i])
            mag_samples = self.mag_max - np.random.exponential(self.mag_ex, total_element)
            for j in range(len(mag_samples)):
                while(mag_samples[j] < 15.75):
                    mag_samples[j] = (self.mag_max - np.random.exponential(self.mag_ex, 1))[0]
                mag_samples[j] = tools.toflux(mag_samples[j])
                if j <= len(galaxy_locs_cluster[i]):
                    mag_samples[j] = mag_samples[j]*(1+self.Z_to_zis(redshift_samples[i], mass_samples[i], 1)[0])
                else:
                    mag_samples[j] = mag_samples[j]*(1+self._sample_redshift_bg())
            flux_samples.append(mag_samples)
        return flux_samples
    
    def _sample_shape(self, galaxy_locs, galaxy_locs_cluster):
        G1_size_samples = []
        G2_size_samples = []
        for i in range(self.size):
            total_element = len(galaxy_locs[i]) + len(galaxy_locs_cluster[i])
            G1_size_samples.append(gennorm.rvs(self.G1_beta, self.G1_loc, self.G1_scale, total_element))
            G2_size_samples.append(gennorm.rvs(self.G2_beta, self.G2_loc, self.G2_scale, total_element))
            for j in range(total_element):
                while G1_size_samples[i][j]**2 + G2_size_samples[i][j]**2 >= 1 or G1_size_samples[i][j] >= 0.8 or G2_size_samples[i][j] >= 0.8:
                    G1_size_samples[i][j] = gennorm.rvs(self.G1_beta, self.G1_loc, self.G1_scale, 1)[0]
                    G2_size_samples[i][j] = gennorm.rvs(self.G2_beta, self.G2_loc, self.G2_scale, 1)[0]
        return G1_size_samples, G2_size_samples
    
    def galaxy_flux_ratio(self, size):
        # ["G", "R", "I", "Z"]
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
    
    def make_catalog(self, flux_samples, t_size_samples, G1_size_samples, G2_size_samples,geo_galaxy, geo_galaxy_cluster, galaxy_locs, galaxy_locs_cluster):
        res = []
        for i in range(len(flux_samples)):
            ratios = self.galaxy_flux_ratio(len(geo_galaxy_cluster[i]) + len(geo_galaxy[i]))
            if len(geo_galaxy_cluster[i]) != 0:
                ratios[:len(geo_galaxy_cluster[i])] = ratios[np.random.randint(0, len(geo_galaxy_cluster[i]) - 1)]
            fluxes = np.array(flux_samples[i])[:, np.newaxis] * np.array(ratios)
            mock_catalog = pd.DataFrame()
            if len(geo_galaxy_cluster[i]) != 0:
                mock_catalog["RA"] = np.append(np.array(geo_galaxy_cluster[i])[:, 0], np.array(geo_galaxy[i])[:, 0])
                mock_catalog["DEC"] = np.append(np.array(geo_galaxy_cluster[i])[:, 1], np.array(geo_galaxy[i])[:, 1])
                mock_catalog["X"] = np.append(np.array(galaxy_locs[i])[:, 0], np.array(galaxy_locs_cluster[i])[:, 0])
                mock_catalog["Y"] = np.append(np.array(galaxy_locs[i])[:, 1], np.array(galaxy_locs_cluster[i])[:, 1])
            else:
                mock_catalog["RA"] = np.array(geo_galaxy[i])[:, 0]
                mock_catalog["DEC"] = np.array(geo_galaxy[i])[:, 1]
                mock_catalog["X"] = np.array(galaxy_locs[i])[:, 0]
                mock_catalog["Y"] = np.array(galaxy_locs[i])[:, 1]
            mock_catalog["FLUX_R"] = fluxes[:, 1]
            mock_catalog["MAG_R"] = -2.5*np.log10(mock_catalog["FLUX_R"]) + 30
            mock_catalog["FLUX_G"] = fluxes[:, 0]
            mock_catalog["MAG_G"] = -2.5*np.log10(mock_catalog["FLUX_G"]) + 30
            mock_catalog["FLUX_I"] = fluxes[:, 2]
            mock_catalog["MAG_I"] = -2.5*np.log10(mock_catalog["FLUX_I"]) + 30
            mock_catalog["FLUX_Z"] = fluxes[:, 3]
            mock_catalog["MAG_Z"] = -2.5*np.log10(mock_catalog["FLUX_Z"]) + 30
            mock_catalog["TSIZE"] = t_size_samples[i]
            mock_catalog["FRACDEV"] = 0
            mock_catalog["G1"] = G1_size_samples[i]
            mock_catalog["G2"] = G2_size_samples[i]
            res.append(mock_catalog)
        return res
    
    def to_tiles(self, center_sample, mass_sample, radius_sample, galaxy_locs_cluster):
        tiles = []
        for i in range(len(mass_sample)):
            temp = {}
            temp["mass"] = mass_sample[i]
            temp["exsit"] = len(galaxy_locs_cluster[i]) == 0
            temp["coordinate"] = center_sample[i]
            temp["radius"] = radius_sample[i]
            temp["canvas"] = [self.width, self.height]
            temp["tiles"] = [self.tiles_width, self.tiles_height]
            tiles.append(temp)
        return tiles
            
    def catalog_render(self, catalogs, start_index, tiles):
        for i in range(len(catalogs)):
            mock_catalog = catalogs[i]
            stds = np.array([2.509813, 5.192254, 8.36335, 15.220351]) / 1.3
            file_name_tag = str(i + start_index)
            for j, band in enumerate(("g", "r", "i")):
                name =  self.folder_path + file_name_tag + "_"+ band
                print(name)
                fr = frame.Frame(mock_catalog.to_records(), band=band, name=name,
                                center=(self.ra_cen, self.dec_cen), noise_std=stds[j], canvas_size=5000, )
                fr.render(nprocess=8) 
            ims_all = []
            for j, band in enumerate(("g", "r", "i")):
                name = self.folder_path + file_name_tag + "_" + band + ".fits"
                tmp = fio.read(name)
                os.remove(name)
                os.remove(self.folder_path + file_name_tag + "_" + band + "_epsf.fits")
                ims_all.append(tmp)
            fig = plt.figure(figsize=(12, 12))
            ax = fig.add_subplot(111)
            factor = 0.001
            scales = np.array([1., 1.2, 2.5]) * factor
            nonlinear = 0.12
            clip = 0
            obs_im = images.get_color_image(ims_all[2],
                                            ims_all[1],
                                            ims_all[0],
                                            nonlinear=nonlinear, clip=clip, scales=scales)  
            print(obs_im.max())
            ax.imshow(obs_im * 2, origin='upper')

            ax.set_xlabel("X [pix]")
            ax.set_ylabel("Y [pix]")
            plt.axis('off')
            fig.savefig(self.folder_path + file_name_tag + ".png", bbox_inches='tight',pad_inches=0)
            plt.close(fig)
            filehandler = open(self.folder_path + file_name_tag + "_catalog.pkl", 'wb') 
            pickle.dump(tiles[i], filehandler)

    def sample(self):
        mass_samples = self._sample_mass()
        redshift_samples = self._sample_redshift()
        radius_samples = self._sample_radius(mass_samples, redshift_samples)
        n_galaxy_cluster = self._sample_n_cluster(mass_samples)
        center_samples = self._sample_center()
        galaxy_cluster_locs = self._sample_cluster_locs(center_samples, radius_samples, n_galaxy_cluster)
        n_galaxy = self._sample_n_galaxy()
        for i in range(self.size):
            n_galaxy[i] = int(n_galaxy[i] - n_galaxy_cluster[i])
        galaxy_locs = self._sample_galaxy_locs(n_galaxy)
        geo_galaxy = self.cartesian2geo(galaxy_locs)
        geo_galaxy_cluster = self.cartesian2geo(galaxy_cluster_locs)
        flux_samples = self._sample_flux(galaxy_locs, galaxy_cluster_locs, redshift_samples, mass_samples)
        TSIZE_samples = self._sample_TSIZE(flux_samples)
        G1_size_samples, G2_size_samples = self._sample_shape(galaxy_locs, galaxy_cluster_locs)
        catalogs = self.make_catalog(flux_samples, TSIZE_samples, G1_size_samples, G2_size_samples,geo_galaxy, geo_galaxy_cluster, galaxy_locs, galaxy_cluster_locs)
        tiles = self.to_tiles(center_samples, mass_samples, radius_samples, galaxy_cluster_locs)
        for i in range(self.threadings):
            x = multiprocessing.Process(target = self.catalog_render, args = (catalogs[i*self.size//self.threadings:(i+1)*self.size//self.threadings], i*self.size//self.threadings, tiles[i*self.size//self.threadings:(i+1)*self.size//self.threadings], ))
            x.start()
        # return catalogs, radius_samples, galaxy_cluster_locs, center_samples
