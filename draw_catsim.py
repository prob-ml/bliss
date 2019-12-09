from WeakLensingDeblending import descwl
import numpy as np
from astropy.table import Table
import galsim


# survey_name = 'LSST'
# bands = ['y', 'z', 'i', 'r', 'g', 'u']
# obs_generator = []
# catalogue_name = "/home/imendoza/deblend/galaxy-net/params/OneDegSq.fits"
# catalogue = Table.read(catalogue_name)

# ToDo: Find average size/maximum size of galaxies to know what stamp size to use.
# can use the get_size function that Sowmya wrote.

def get_size(catalog, pixel_scale, i_obs_cond):
    """Returns a astropy.table.column with the size of the galaxy.
    Galaxy size is estimated as second moments size (r_sec) computed as
    described in A1 of Chang et.al 2012. The PSF second moment size, psf_r_sec,
    is computed by galsim from the psf model in obs_cond in the i band.
    The object size is the defined as sqrt(r_sec**2 + 2*psf_r_sec**2).
    Args:
        Args: Class containing input parameters.
        catalog: Catalog with entries corresponding to one blend.
        i_obs_cond: `descwl.survey.Survey` class describing
            observing conditions in i band.
    Returns:
        `astropy.table.Column`s: size of the galaxy.
    """
    f = catalog['fluxnorm_bulge'] / (catalog['fluxnorm_disk'] + catalog['fluxnorm_bulge'])
    hlr_d = np.sqrt(catalog['a_d'] * catalog['b_d'])
    hlr_b = np.sqrt(catalog['a_b'] * catalog['b_b'])
    r_sec = np.hypot(hlr_d * (1 - f) ** 0.5 * 4.66,
                     hlr_b * f ** 0.5 * 1.46)
    psf = i_obs_cond.psf_model
    psf_r_sec = psf.calculateMomentRadius()
    size = np.sqrt(r_sec ** 2 + psf_r_sec ** 2) / pixel_scale
    return Column(size, name='size')


def get_params():
    params = dict(
        survey_name='LSST',
        catalog_name="/home/imendoza/deblend/galaxy-net/params/OneDegSq.fits",
        bands=['y', 'z', 'i', 'r', 'g', 'u'],
    )

    return params


def create_obs(params):
    obs = []
    for band in params['bands']:
        # dictionary of default values.
        survey_dict = descwl.survey.Survey.get_defaults(
            survey_name=params['survey_name'], filter_band=band)
        survey_dict['image_width'] = params['stamp_size'] / survey_dict['pixel_scale']  # pixels
        survey_dict['image_height'] = params['stamp_size'] / survey_dict['pixel_scale']

        descwl_survey = descwl.survey.Survey(no_analysis=True,
                                             survey_name=params['survey_name'],
                                             filter_band=band, **survey_dict)
        obs.append(descwl_survey)

    return obs


def single_band(entry, band, single_obs, params, no_psf=True, draw_method='auto', snr=None):
    """
    Builds galaxy from a single entry in the catalogue.
    :param entry:
    :param obs_cond:
    :return:
    """

    # random deviation from exactly in center of center pixel.
    entry['ra'], entry['dec'] = (np.random.rand(2) - 0.5) * single_obs.pixel_scale

    galaxy_builder = descwl.model.GalaxyBuilder(
        single_obs, no_disk=False, no_bulge=False,
        no_agn=False, verbose_model=False)
    galaxy = galaxy_builder.from_catalog(entry,
                                         entry['ra'],
                                         entry['dec'],
                                         band)

    iso_render_engine = descwl.render.Engine(
        survey=single_obs,
        min_snr=params['min_snr'],
        truncate_radius=30,
        no_margin=False,
        verbose_render=False)

    try:
        iso_render_engine.render_galaxy(
            galaxy, variations_x=None, variations_s=None, variations_g=None,
            no_fisher=True, calculate_bias=False, no_analysis=True, no_psf=no_psf, draw_method=draw_method)

    # Deal with not visible sources.
    except descwl.render.SourceNotVisible:
        if params['verbose']:
            print("Source not visible")
        raise RuntimeError()

    image_temp = galsim.Image(single_obs.image_width, single_obs.image_height)
    image_temp += single_obs.image

    if params['add_noise']:
        generator = galsim.random.BaseDeviate(
            seed=np.random.randint(99999999))
        noise = galsim.PoissonNoise(
            rng=generator,
            sky_level=single_obs.mean_sky_level)

        if snr:
            image_temp.addNoiseSNR(noise, snr=snr, preserve_flux=True)
        else:
            image_temp.addNoise(noise)

    return image_temp
