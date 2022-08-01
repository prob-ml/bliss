from pathlib import Path
from hydra.utils import instantiate

from case_studies.coadds.coadd_decoder import (
    CoaddSingleGalaxyDecoder,
    CoaddUniformGalsimGalaxiesPrior,
)


def get_coadds_cfg(overrides, devices):
    overrides.update({"gpus": devices.gpus, "paths.root": Path(__file__).parents[3].as_posix()})
    overrides = [f"{k}={v}" if v is not None else f"{k}=null" for k, v in overrides.items()]
    with initialize(config_path="../../../case_studies/coadds/config"):
        cfg = compose("config", overrides=overrides)
    return cfg


def test_coadd_prior(get_coadds_cfg, devices):
    pixel_scale = 0.393
    max_n_sources = 1
    max_shift = 0.5
    num_dithers = 4
    cfg = get_coadds_cfg({}, devices)
    prior = instantiate(cfg.datasets.sdss_galaxies_coadd.prior)

    CoaddUniformGalsimGalaxiesPrior(prior, max_n_sources, max_shift, num_dithers).sample(
        num_dithers
    )
