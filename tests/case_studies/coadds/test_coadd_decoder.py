from pathlib import Path

from hydra import compose, initialize
from hydra.utils import instantiate

from case_studies.coadds.coadd_decoder import CoaddUniformGalsimGalaxiesPrior


def get_coadds_cfg(get_config, devices):
    with initialize(config_path="../case_studies/coadds/config"):
        cfg = compose("config")
    return cfg


def test_coadd_prior(overrides, devices):
    max_n_sources = 1
    max_shift = 0.5
    num_dithers = 4
    cfg = get_coadds_cfg(get_config, devices)
    prior = instantiate(cfg.datasets.sdss_galaxies_coadd.prior)

    CoaddUniformGalsimGalaxiesPrior(prior, max_n_sources, max_shift, num_dithers).sample(
        num_dithers
    )
