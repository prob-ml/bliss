from hydra.utils import instantiate

from case_studies.coadds.coadd_decoder import (
    CoaddGalsimBlends,
    CoaddSingleGalaxyDecoder,
    CoaddUniformGalsimGalaxiesPrior,
)


def test_coadd_prior(get_coadds_config, devices):
    max_n_sources = 1
    max_shift = 0.5
    num_dithers = 4
    cfg = get_coadds_config({}, devices)
    prior = instantiate(cfg.datasets.sdss_galaxies_coadd.prior)

    CoaddUniformGalsimGalaxiesPrior(prior, max_n_sources, max_shift, num_dithers).sample(
        num_dithers
    )


def test_coadd_single_decoder(get_coadds_config, devices):
    cfg = get_coadds_config({}, devices)
    decoder = instantiate(cfg.datasets.sdss_galaxies_coadd.decoder)

    CoaddSingleGalaxyDecoder(
        decoder,
        decoder.n_bands,
        decoder.pixel_scale,
        "./data/sdss/psField-000094-1-0012-PSF-image.npy",
    )


def test_coadd_galsim_blend(get_coadds_config, devices):
    cfg = get_coadds_config({}, devices)
    prior = instantiate(cfg.datasets.sdss_galaxies_coadd.prior)
    decoder = instantiate(cfg.datasets.sdss_galaxies_coadd.decoder)
    background = instantiate(cfg.datasets.galsim_blended_galaxies_coadd.background)

    max_tile_n_sources = 1
    tile_slen = 4
    num_workers = 5
    batch_size = 1000
    n_batches = 1
    dithers = [((-0.5 - 0.5) * torch.rand((2,)) + 0.5).numpy() for x in range(3)]

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
