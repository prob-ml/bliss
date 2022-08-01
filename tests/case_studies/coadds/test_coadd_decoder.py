import os

from hydra import compose, initialize
from hydra.utils import instantiate

from case_studies.coadds.coadd_decoder import (
    CoaddSingleGalaxyDecoder,
    CoaddUniformGalsimGalaxiesPrior,
)

os.chdir("/bliss")
with initialize(config_path="./case_studies/coadds/config"):
    cfg = compose("config", overrides=[])
prior = instantiate(cfg.datasets.sdss_galaxies_coadd.prior)
decoder = instantiate(cfg.datasets.sdss_galaxies_coadd.decoder)


def test_coadd_prior(prior):
    max_n_sources = 1
    max_shift = 0.5
    num_dithers = 4
    CoaddUniformGalsimGalaxiesPrior(prior, max_n_sources, max_shift, num_dithers).sample(
        num_dithers
    )


def test_coadd_single_decoder(prior):
    max_n_sources = 1
    max_shift = 0.5
    num_dithers = 4
    sampled_cuggp = CoaddUniformGalsimGalaxiesPrior(
        prior, max_n_sources, max_shift, num_dithers
    ).sample(num_dithers)
    galaxy_params = sampled_cuggp["galaxy_params"]
    dithers = sampled_cuggp["dithers"]
    offset = None

    csgd = CoaddSingleGalaxyDecoder(
        decoder,
        decoder.n_bands,
        decoder.pixel_scale,
        "./data/sdss/psField-000094-1-0012-PSF-image.npy",
    )
    csgd.render_galaxy(
        galaxy_params=galaxy_params[0],
        slen=decoder.slen,
        psf=decoder.psf,
        offset=offset,
        dithers=dithers,
    )
