import galsim
import numpy as np
import scipy.stats as stats
from torch.utils.data import Dataset
from astropy.table import Table
import os
import draw_catsim
from WeakLensingDeblending import descwl

os.chdir("/home/imendoza/deblend/galaxy-net")


class CatsimGalaxies(Dataset):

    def __init__(self, survey_name=None, image_size=40, filter_dict=None, fixed_size=False, snr=200, bands=None,
                 **render_kwargs):
        """
        This class reads a random entry from the OneDegSq.fits file (sample from the Catsim catalogue) and returns a
        galaxy drawn from the catalogue with realistic seeing conditions using functions from WeakLensingDeblending.

        For now, only one galaxy can be returned at once.

        :param snr: The SNR of the galaxy to draw, if None uses the actually seeing SNR from LSST survey.
        :param render_kwargs: Additional keyword arguments that will go into the renderer.
        :param filter_dict: Exclude some entries from based CATSIM on dict of filters, default is to exclude >=25.3 i_ab.
        :param stamp_size: In arcsecs.
        """

        assert survey_name is None, "Only using default survey name for now = LSST"
        assert image_size >= 40, "Does not seem to work well if the number of pixels is too low."
        assert filter_dict is None, "Not supporting different dict yet, need to change argparse + save_props below."
        assert bands is None, "Only using default number of bands = 6 for now."

        params = draw_catsim.get_default_params()
        self.survey_name = params['survey_name'] if not survey_name else survey_name
        self.bands = params['bands'] if not bands else bands
        self.image_size = image_size
        self.pixel_scale = descwl.survey.Survey.get_defaults(self.survey_name, '*')['pixel_scale']
        self.stamp_size = self.pixel_scale * self.image_size  # arcsecs.
        self.snr = snr

        self.fixed_size = fixed_size

        self.renderer = draw_catsim.Render(self.survey_name, self.bands, self.stamp_size, self.pixel_scale,
                                           snr=self.snr, **render_kwargs)

        self.table = Table.read(params['catalog_name'])
        self.table = self.table[np.random.permutation(len(self.table))]  # shuffle in case that order matters.
        self.filtered_dict = CatsimGalaxies.get_default_filters() if filter_dict is None else filter_dict
        self.cat = self.get_filtered_table()

        self.prepare_cat()

    def __len__(self):
        return len(self.cat)

    # ToDo: Remove all non-visible sources.
    def __getitem__(self, idx):

        while True:  # loop until visible galaxy is selected.
            try:
                entry = self.cat[idx]
                final, background = self.renderer.draw(entry)
                break

            except descwl.render.SourceNotVisible:
                idx = np.random.choice(np.arange(len(self)))  # select some other random galaxy to return.

        return {'image': final,
                'background': background,
                'num_galaxies': 1}

    def print_props(self, prop_file):
        print(f"snr: {self.snr} \n"
              f"fixed size: {self.fixed_size} \n"
              f"survey name: {self.survey_name} \n"
              f"bands: {self.bands}\n"
              f"pixel scale: {self.pixel_scale}\n"
              f"filter_dict: Default\n"
              f"min_snr: {self.renderer.min_snr}\n"
              f"truncate_radius: {self.renderer.truncate_radius}\n"
              f"add_noise: {self.renderer.add_noise}\n"
              f"preserve_flux: {self.renderer.preserve_flux}\n",
              file=prop_file)

    def prepare_cat(self):
        # random deviation from exactly in center of center pixel, in arcsecs.
        self.cat['ra'] = (np.random.rand(len(self.cat)) - 0.5) * self.pixel_scale  # arcsecs
        self.cat['dec'] = (np.random.rand(len(self.cat)) - 0.5) * self.pixel_scale

        if self.fixed_size:
            for i, _ in enumerate(self.cat):
                self.cat[i] = self.fix_size(self.cat[i])

    def get_filtered_table(self):
        cat = self.table.copy()
        for param, pfilter in self.filtered_dict.items():
            cat = cat[pfilter(param, cat)]
        return cat

    def fix_size(self, entry):
        hlr_d = None
        if entry['a_d'] != 0.0 and entry['b_d'] != 0.0:
            hlr_d_old = np.sqrt(entry['b_d'] * entry['a_d'])
            q_d = entry['b_d'] / entry['a_d']
            hlr_d = self.stamp_size / 15
            a_d = hlr_d / np.sqrt(q_d)
            b_d = hlr_d * np.sqrt(q_d)
            entry['a_d'] = a_d
            entry['b_d'] = b_d

        if entry['a_b'] != 0.0 and entry['b_b'] != 0.0:

            if hlr_d is not None:
                hlr_b_old = np.sqrt(entry['b_b'] * entry['a_b'])
                hlr_b = hlr_d * hlr_b_old / hlr_d_old
            else:
                hlr_b = self.stamp_size / 15
            q_b = entry['b_b'] / entry['a_b']
            entry['a_b'] = hlr_b / np.sqrt(q_b)
            entry['b_b'] = hlr_b * np.sqrt(q_b)

        return entry

    @staticmethod
    def get_default_filters():
        filters = dict(
            i_ab=lambda param, x: x[param] <= 25.3  # cut on magnitude same as BTK does.
        )
        return filters


class GalBasic(Dataset):

    def __init__(self, slen, num_images=None, survey_name='lsst',
                 snr=200, sky=700, flux=None, num_galaxies=1):
        """
        This class uses and returns a random Gaussian Galaxy, the flux is adjusted based on slen and sky so that the
        ratio from image to background is approximately 0.30 like in Jeff's original code.

        There is always only oen galaxy and it is always randomly located somewhere in the center pixel.
        """
        super(GalBasic, self).__init__()  # runs init of the super class.

        assert slen >= 40, "Does not seem to work well if the number of pixels is too low."

        self.slen = slen  # number of pixel dimensions.
        self.snr = snr
        self.centered = True
        self.num_bands = 1
        self.num_galaxies = num_galaxies
        self.sky = sky
        self.pixel_scale = 0.2
        self.num_images = num_images

        # adjust flux depending on size and same ratio as Jeff.
        if flux is None:
            self.flux = 5e4 * (self.slen / 15) ** 2
        else:
            self.flux = flux

        # do not want too small of sigma to avoid pixel galaxies.
        # we can use the same size for everything for now.
        self.sigma = self.slen * self.pixel_scale / 8

        if self.num_bands > 1 or not self.centered or survey_name != 'lsst':
            raise NotImplementedError("Not yet implemented multiple bands, not centered galaxy, "
                                      "not lsst survey, or more than one galaxy.")

    def __len__(self):
        return self.num_images  # meaningless

    def __getitem__(self, idx):
        """
        Right now this completely ignores the index and returns some random Gaussian galaxy using galsim.

        :param idx:
        :return:
        """

        # get random galaxy parameters.
        r = min(np.random.sample(), 0.99)  # magnitude needs to be < 1 .
        theta = np.random.sample() * np.pi * 2
        e1, e2 = max(min(r * np.cos(theta), 0.4), -0.4), max(min(r * np.sin(theta), 0.4), -0.4)
        loc = np.random.rand(2) - 0.5

        gal = galsim.Gaussian(flux=self.flux, sigma=self.sigma)
        gal = gal.shear(e1=e1, e2=e2)
        gal = gal.shift(dx=loc[0], dy=loc[1])  # randomly somewhere in the center pixel.
        img = gal.drawImage(nx=self.slen, ny=self.slen, scale=self.pixel_scale,
                            method='auto')

        noisy_img = img.copy()

        # add noise.
        rng = galsim.BaseDeviate(0)
        noise = galsim.PoissonNoise(rng=rng, sky_level=self.sky)
        _ = noisy_img.addNoiseSNR(noise, self.snr, preserve_flux=True)

        # obtain background
        noisy_arr = noisy_img.array
        image = np.zeros((self.num_bands, self.slen, self.slen), dtype=np.float32)
        image[0, :, :] = noisy_arr
        background = np.full_like(image, self.sky)
        image += background

        return {'image': image,
                'background': background,
                'num_galaxies': 1}

    def print_props(self, prop_file):
        pass


class Synthetic(Dataset):

    def __init__(self, slen, mean_galaxies=2, min_galaxies=0, max_galaxies=3,
                 num_bands=5, padding=3, centered=False, num_images=1000,
                 flux=30000):
        """
        Questions:
        - What is s_density?
        - What  is A**B where both are matrices?
        """
        super(Synthetic, self).__init__()

        self.slen = slen  # number of pixel dimensions.
        self.mean_galaxies = mean_galaxies
        self.min_galaxies = min_galaxies
        self.max_galaxies = max_galaxies
        self.num_images = num_images
        self.num_bands = num_bands
        self.padding = padding
        self.centered = centered
        self.flux = flux
        self.sky = 700
        self.snr = -1

    def __len__(self):
        return self.num_images  # meaningless.

    def __getitem__(self, idx):
        # right now this completely ignores the index.

        background_sample = np.random.randn(self.num_bands, self.slen, self.slen) * np.sqrt(self.sky) + self.sky
        background = np.full_like(background_sample, self.sky, dtype=np.float32)
        image = np.asarray(background_sample, dtype=np.float32)

        axis_lengths = np.array([[3.0, 0.0], [0.0, 7.0]])

        poisson_galaxies = np.random.poisson(self.mean_galaxies)
        num_galaxies = np.maximum(np.minimum(poisson_galaxies, self.max_galaxies), self.min_galaxies)

        for j in range(num_galaxies):
            angle = np.pi * np.random.rand()
            rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
            covar = np.matmul(np.matmul(rotation.transpose(), axis_lengths),
                              rotation)  # rotate the initial axis lengths
            loc = np.random.rand(2) - 0.5  # centered by default.
            if not self.centered:
                loc *= self.slen - 2 * self.padding
            bvn = stats.multivariate_normal(loc, covar)

            offset = (self.slen - 1) / 2
            y, x = np.mgrid[-offset:(offset + 1), -offset:(offset + 1)]
            pos = np.dstack((y, x))  # all the positions of the different pixels in the considered grid.
            s_density = bvn.pdf(pos)

            temperature = 1.0 + np.random.rand()
            flux = self.flux * temperature ** np.mgrid[1:(self.num_bands + 1)]
            flux = flux.reshape(self.num_bands, 1, 1)
            s_density = s_density.reshape(1, self.slen, self.slen)
            s_intensity = s_density * flux
            s_noise = np.random.randn(self.num_bands, self.slen, self.slen)
            s_contrib = s_intensity + np.sqrt(s_intensity) * s_noise
            image += s_contrib

        return {'image': image,
                'background': background,
                'num_galaxies': num_galaxies}

    def print_props(self, prop_file):
        pass


class CatsimData(Dataset):

    def __init__(self):
        """
        This class reads the relevant parameters OneDegSq.fits file and returns samples from this
        ~800k row matrix.
        """
        super(CatsimData, self).__init__()

        # pa_disk = pa_bulge (by assumption)
        self.param_names = ['redshift',
                            'fluxnorm_bulge', 'fluxnorm_disk', 'fluxnorm_agn',
                            'a_b', 'a_d', 'b_b', 'b_d', 'pa_disk',
                            'u_ab', 'g_ab', 'r_ab', 'i_ab', 'z_ab', 'y_ab']
        self.num_params = len(self.param_names)

        self.table = Table.read("/home/imendoza/deblend/galaxy-net/params/OneDegSq.fits")
        np.random.shuffle(self.table)  # shuffle just in case order of galaxies matters in original table.
        self.params = self.table[self.param_names]  # array of tuples of len = 18.

    def __len__(self):
        return self.table.shape[0]

    def __getitem__(self, idx):
        return np.array([self.params[idx][i] for i in range(len(self.param_names))], dtype=np.float32)
