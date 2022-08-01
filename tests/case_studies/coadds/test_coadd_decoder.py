from hydra.utils import instantiate

from case_studies.coadds.coadd_decoder import CoaddUniformGalsimGalaxiesPrior


def test_coadd_prior(get_coadds_config, devices):
    max_n_sources = 1
    max_shift = 0.5
    num_dithers = 4
    cfg = get_coadds_config({}, devices)
    prior = instantiate(cfg.datasets.sdss_galaxies_coadd.prior)

    CoaddUniformGalsimGalaxiesPrior(prior, max_n_sources, max_shift, num_dithers).sample(
        num_dithers
    )
