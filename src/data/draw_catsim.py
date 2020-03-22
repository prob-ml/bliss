from packages.WeakLensingDeblending import descwl
import numpy as np
from astropy.table import Column
import galsim

from GalaxyModel.src.utils import const


def get_default_params():
    params = dict(
        survey_name='LSST',
        catalog_file_path=const.data_path.joinpath("raw/OneDegSq.fits"),
        bands=['y', 'z', 'i', 'r', 'g', 'u'],
    )

    return params


# ToDo: LATER More flexibility than drawing randomly centered in central pixel.
class Render(object):

    def __init__(self, survey_name, bands, stamp_size, pixel_scale, snr=None, dtype=None,
                 min_snr=0.05, truncate_radius=30, add_noise=True, preserve_flux=True,
                 verbose=False):
        """
        Can draw a single entry in CATSIM in the given bands.

        NOTE: Background is constant given the band, survey_name, image size, and default survey_dict, so it can
        be obtained in advance only once.
        """
        self.survey_name = survey_name
        self.bands = bands
        self.num_bands = len(self.bands)
        self.stamp_size = stamp_size  # arcsecs
        self.pixel_scale = pixel_scale
        self.image_size = int(self.stamp_size / self.pixel_scale)  # pixels.
        self.snr = snr
        self.min_snr = min_snr
        self.truncate_radius = truncate_radius
        self.add_noise = add_noise
        self.preserve_flux = preserve_flux  # when changing SNR.
        self.verbose = verbose
        self.dtype = dtype

        self.obs = self.get_obs()
        self.background = self.get_background()

    # @profile
    def get_obs(self):
        """
        Returns a list of :class:`Survey` objects, each of them has an image attribute which is
        where images are written to by iso_render_engine.render_galaxy.
        :return:
        """
        obs = []
        for band in self.bands:
            # dictionary of default values.
            survey_dict = descwl.survey.Survey.get_defaults(
                survey_name=self.survey_name, filter_band=band)

            assert self.pixel_scale == survey_dict['pixel_scale'], "Pixel scale does not match particular band?"
            survey_dict['image_width'] = self.image_size  # pixels
            survey_dict['image_height'] = self.image_size

            descwl_survey = descwl.survey.Survey(no_analysis=True,
                                                 survey_name=self.survey_name,
                                                 filter_band=band, **survey_dict)
            obs.append(descwl_survey)

        return obs

    def get_background(self):
        background = np.zeros((self.num_bands, self.image_size, self.image_size), dtype=self.dtype)
        for i, single_obs in enumerate(self.obs):
            background[i, :, :] = single_obs.mean_sky_level
        return background

    # ToDo: Consider a context manager.
    def reset_obs(self):
        """
        Reset it so we can draw on it again.
        :return:
        """
        for single_obs in self.obs:
            single_obs.image = galsim.Image(bounds=single_obs.image_bounds, scale=self.pixel_scale,
                                            dtype=np.float32)

    # noinspection DuplicatedCode
    def get_size(self, cat):
        """
        Return a astropy.Column with the size of each entry of the catalog given the current
        rendering observing conditions / survey.
        """
        assert 'i' in self.bands, "Requires the i band to be present."
        i_obs = self.obs[self.bands.index('i')]

        f = cat['fluxnorm_bulge'] / (cat['fluxnorm_disk'] + cat['fluxnorm_bulge'])
        hlr_d = np.sqrt(cat['a_d'] * cat['b_d'])
        hlr_b = np.sqrt(cat['a_b'] * cat['b_b'])
        r_sec = np.hypot(hlr_d * (1 - f) ** 0.5 * 4.66,
                         hlr_b * f ** 0.5 * 1.46)
        psf = i_obs.psf_model
        psf_r_sec = psf.calculateMomentRadius()
        size = np.sqrt(r_sec ** 2 + psf_r_sec ** 2) / self.pixel_scale  # size is in pixels.
        return Column(size, name='size')

    def draw(self, entry):
        """
        Return a multi-band image corresponding to the entry from the catalog given.

        * The final image includes its background based on survey's sky level.
        * Each galaxy is aligned across the bands because `entry` is the same.
        """
        image = np.zeros((self.num_bands, self.image_size, self.image_size), dtype=self.dtype)

        for i, band in enumerate(self.bands):
            image_no_background = self.single_band(entry, self.obs[i], band)
            image[i, :, :] = image_no_background + self.background[i]

        self.reset_obs()

        return image, self.background

    def single_band(self, entry, single_obs, band):
        """
        Builds galaxy from a single entry in the catalog. With no background sky level added.
        :param entry:
        :param single_obs:
        :param band:
        :return:
        """

        galaxy_builder = descwl.model.GalaxyBuilder(single_obs, no_disk=False, no_bulge=False,
                                                    no_agn=False, verbose_model=False)

        galaxy = galaxy_builder.from_catalog(entry, entry['ra'], entry['dec'], band)

        iso_render_engine = descwl.render.Engine(
            survey=single_obs,
            min_snr=self.min_snr,
            truncate_radius=self.truncate_radius,
            no_margin=False,
            verbose_render=False)

        # Up to this point, single_obs has not been changed by the previous 3 statements.

        try:
            # this line draws the given galaxy image onto single_obs.image, this is the only change in single_obs.
            iso_render_engine.render_galaxy(
                galaxy, variations_x=None, variations_s=None, variations_g=None,
                no_fisher=True, calculate_bias=False, no_analysis=True)  # saves image in single_obs

        except descwl.render.SourceNotVisible:
            if self.verbose:
                print("Source not visible")
            raise descwl.render.SourceNotVisible  # pass it on with a warning.

        image_temp = galsim.Image(self.image_size, self.image_size)
        image_temp += single_obs.image

        if self.add_noise:
            generator = galsim.random.BaseDeviate(
                seed=np.random.randint(99999999))
            noise = galsim.PoissonNoise(
                rng=generator,
                sky_level=single_obs.mean_sky_level)  # remember PoissonNoise assumes background already subtracted off.

            # Both of the adding noise methods add noise on the image consisting of the (galaxy flux + background), but
            # then remove the background at the end so we need to add it later.
            if self.snr:
                image_temp.addNoiseSNR(noise, snr=self.snr, preserve_flux=self.preserve_flux)
            else:
                image_temp.addNoise(noise)

        return image_temp.array
