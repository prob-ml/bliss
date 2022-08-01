from hydra.utils import instantiate

from case_studies.coadds.coadd_decoder import CoaddUniformGalsimGalaxiesPrior


def test_coadd_prior(get_coadds_config, devices):
    max_n_sources = 1
    max_shift = 0.5
    num_dithers = 4
    cfg = get_coadds_config({}, devices)
    prior = instantiate(cfg.datasets.sdss_galaxies_coadd.prior)

    return CoaddUniformGalsimGalaxiesPrior(prior, max_n_sources, max_shift, num_dithers).sample(
        num_dithers
    )


def test_coadd_single_decoder(get_config, devices):
    sampled_cuggp = test_coadd_prior(get_config, devices)
    galaxy_params = sampled_cuggp["galaxy_params"]
    dithers = sampled_cuggp["dithers"]
    offset = None
    offset = None

    cfg = get_config({}, devices)
    decoder = instantiate(cfg.datasets.sdss_galaxies_coadd.decoder)

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
