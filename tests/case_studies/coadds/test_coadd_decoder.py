import torch
from hydra.utils import instantiate

from case_studies.coadds.coadd_decoder import (
    CoaddGalsimBlends,
    CoaddSingleGalaxyDecoder,
    CoaddUniformGalsimGalaxiesPrior,
    FullCatalogDecoder,
)


def test_coadd_prior(get_config, devices):
    max_n_sources = 1
    max_shift = 0.5
    num_dithers = 4
    cfg = get_config({}, devices)
    prior = instantiate(cfg.datasets.sdss_galaxies_coadd.prior)

    CoaddUniformGalsimGalaxiesPrior(prior, max_n_sources, max_shift, num_dithers).sample(
        num_dithers
    )


def test_coadd_single_decoder(get_config, devices):
    sampled_cuggp = test_coadd_prior(get_config, devices)
    galaxy_params = sampled_cuggp["galaxy_params"]
    dithers = sampled_cuggp["dithers"]
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


def test_full_catalog_decoder(get_config, devices):
    cfg = get_config({}, devices)
    decoder = instantiate(cfg.datasets.galsim_blended_galaxies_coadd.decoder)
    slen = 40
    bp = 24
    dithers = [((-0.5 - 0.5) * torch.rand((2,)) + 0.5).numpy() for x in range(3)]
    FullCatalogDecoder(decoder, slen, bp, dithers)


def test_coadd_galsim_blend(get_config, devices):
    cfg = get_config({}, devices)
    decoder = instantiate(cfg.datasets.galsim_blended_galaxies_coadd.decoder)
    prior = instantiate(cfg.datasets.galsim_blended_galaxies_coadd.prior)
    decoder = instantiate(cfg.datasets.galsim_blended_galaxies_coadd.decoder)
    background = instantiate(cfg.datasets.galsim_blended_galaxies_coadd.background)

    tile_slen = 4
    max_tile_n_sources = 1
    num_workers = 5
    batch_size = 1000
    n_batches = 1

    CoaddGalsimBlends(
        prior=prior,
        decoder=decoder,
        background=background,
        tile_slen=tile_slen,
        max_sources_per_tile=max_tile_n_sources,
        num_workers=num_workers,
        batch_size=batch_size,
        n_batches=n_batches,
    )
