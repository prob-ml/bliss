from WeakLensingDeblending import descwl
import numpy as np
from astropy.table import Table
import galsim

survey_name = 'LSST'
bands = ['y', 'z', 'i', 'r', 'g', 'u']
obs_generator = []
catalogue_name = "/home/imendoza/deblend/galaxy-net/params/OneDegSq.fits"

catalogue = Table.read(catalogue_name)

params = dict(
    catalogue_name=catalogue_name,
    bands=['y', 'z', 'i', 'r', 'g', 'u'],
    stamp_size=8,  # arcsecs
    add_noise=True,
    seed=0,
    min_snr=0.05,
)


# ToDo: Find average size/maximum size of galaxies to know what stamp size to use.
# can use the get_size function that Sowmya wrote.

def create_obs(params):
    obs = []
    for band in params.bands:
        # dictionary of default values.
        survey_dict = descwl.survey.Survey.get_defaults(
            survey_name=survey_name, filter_band='r')
        survey_dict['image_width'] = params.stamp_size / survey_dict['pixel_scale']
        survey_dict['image_height'] = params.stamp_size / survey_dict['pixel_scale']

        descwl_survey = descwl.survey.Survey(no_analysis=True,
                                             survey_name=survey_name,
                                             filter_band=band, **survey_dict)
        obs.append(descwl_survey)

    return obs


def single_band(entry, band, single_obs, params, no_psf=True):
    """
    Builds galaxy from a single entry in the catalogue.
    :param entry:
    :param obs_cond:
    :return:
    """

    # ToDo: Adjust entry so that all of galaxies are centered at dec, ra = 0.
    galaxy_builder = descwl.model.GalaxyBuilder(
        single_obs, no_disk=False, no_bulge=False,
        no_agn=False, verbose_model=False)
    stamp_size = np.int(params.stamp_size / params.pixel_scale)
    galaxy = galaxy_builder.from_catalog(entry,
                                         entry['ra'],
                                         entry['dec'],
                                         band)

    # ToDo: draw without the PSF.
    iso_render_engine = descwl.render.Engine(
        survey=single_obs,
        min_snr=params.min_snr,
        truncate_radius=30,
        no_margin=False,
        verbose_render=False)
    iso_render_engine.render_galaxy(
        galaxy, variations_x=None, variations_s=None, variations_g=None,
        no_fisher=True, calculate_bias=False, no_analysis=True)

    image_temp = galsim.Image((stamp_size, stamp_size))
    image_temp += single_obs.image

    if Args.add_noise:
        generator = galsim.random.BaseDeviate(
            seed=np.random.randint(99999999))
        noise = galsim.PoissonNoise(
            rng=generator,
            sky_level=single_obs.mean_sky_level)
        image_temp.addNoise(noise)

    # ToDo: Deal with not visible sources.

    # except descwl.render.SourceNotVisible:
    #     if Args.verbose:
    #         print("Source not visible")
    #     blend_catalog['not_drawn_' + band][k] = 1
    #     continue

    return image_temp

# self.sky = self.survey.mean_sky_level
