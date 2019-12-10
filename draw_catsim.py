from WeakLensingDeblending import descwl
import numpy as np
from astropy.table import Column
import galsim


def get_default_params():
    params = dict(
        survey_name='LSST',
        catalog_name="/home/imendoza/deblend/galaxy-net/params/OneDegSq.fits",
        bands=['y', 'z', 'i', 'r', 'g', 'u'],
    )

    return params


class Render(object):

    def __init__(self, survey_name, bands, stamp_size, snr=None, no_psf=None, draw_method='auto',
                 min_snr=0.05, truncate_radius=30, add_noise=True, preserve_flux=True,
                 verbose=False):
        """
        Knows how to draw a single entry in CATSIM in the given bands.
        """
        self.survey_name = survey_name
        self.bands = bands
        self.num_bands = len(self.bands)
        self.stamp_size = stamp_size  # arcsecs
        self.pixel_scale = descwl.survey.Survey.get_defaults(survey_name, '*')['pixel_scale']
        self.image_size = int(self.stamp_size / self.pixel_scale)  # pixels.
        self.snr = snr
        self.min_snr = min_snr
        self.truncate_radius = truncate_radius
        self.no_psf = no_psf
        self.add_noise = add_noise
        self.draw_method = draw_method
        self.preserve_flux = preserve_flux  # when changing SNR.
        self.verbose = verbose

    def get_obs(self):
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

    def get_size(self, cat):
        """
        Return the size of the catalog given the current rendering observing conditions / survey.
        """
        obs = self.get_obs()
        assert 'i' in self.bands, "Requires the i band to be present."
        i_obs = obs[self.bands.index('i')]

        f = cat['fluxnorm_bulge'] / (cat['fluxnorm_disk'] + cat['fluxnorm_bulge'])
        hlr_d = np.sqrt(cat['a_d'] * cat['b_d'])
        hlr_b = np.sqrt(cat['a_b'] * cat['b_b'])
        r_sec = np.hypot(hlr_d * (1 - f) ** 0.5 * 4.66,
                         hlr_b * f ** 0.5 * 1.46)
        psf = i_obs.psf_model
        psf_r_sec = psf.calculateMomentRadius()
        size = np.sqrt(r_sec ** 2 + psf_r_sec ** 2) / self.pixel_scale
        return Column(size, name='size')

    def draw(self, entry):
        """
        Return a multi-band image corresponding to the entry from the catalogue given.
        The final image includes its background based on survey's sky level.
        """
        obs = self.get_obs()
        final = np.zeros((len(self.bands), self.image_size, self.image_size), dtype=np.float32)
        backs = np.zeros((len(self.bands), self.image_size, self.image_size), dtype=np.float32)

        for i, band in enumerate(self.bands):
            image = self.single_band(entry, obs[i], band)
            background = self.get_background(obs[i])
            final[i, :, :] = image + background  # final image includes background.
            backs[i, :, :] = background

        return final, backs

    def get_background(self, single_obs):
        background = np.zeros((self.image_size, self.image_size), dtype=np.float32)
        background[:, :] = single_obs.mean_sky_level
        return background

    # ToDo: More flexibility than drawing exactly centered.
    def single_band(self, entry, single_obs, band):
        """
        Builds galaxy from a single entry in the catalogue.
        :param entry:
        :param single_obs:
        :param band:
        :return:
        """

        # random deviation from exactly in center of center pixel, in arcsecs.
        entry['ra'], entry['dec'] = (np.random.rand(2) - 0.5) * self.pixel_scale

        galaxy_builder = descwl.model.GalaxyBuilder(single_obs, no_disk=False, no_bulge=False,
                                                    no_agn=False, verbose_model=False)

        galaxy = galaxy_builder.from_catalog(entry, entry['ra'], entry['dec'], band)

        iso_render_engine = descwl.render.Engine(
            survey=single_obs,
            min_snr=self.min_snr,
            truncate_radius=self.truncate_radius,
            no_margin=False,
            verbose_render=False)

        try:
            iso_render_engine.render_galaxy(
                galaxy, variations_x=None, variations_s=None, variations_g=None,
                no_fisher=True, calculate_bias=False, no_analysis=True, no_psf=self.no_psf,
                draw_method=self.draw_method)

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

            if self.snr:
                image_temp.addNoiseSNR(noise, snr=self.snr, preserve_flux=self.preserve_flux)
            else:
                image_temp.addNoise(noise)

        return image_temp.array
