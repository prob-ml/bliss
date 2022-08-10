from hydra.utils import instantiate
import torch

from case_studies.coadds.coadd_decoder import (
    CoaddGalsimBlends,
    CoaddSingleGalaxyDecoder,
    CoaddUniformGalsimGalaxiesPrior,
)


def test_coadd_galsim_blend(get_coadds_config, devices):
    cfg = get_coadds_config({}, devices)
    prior = instantiate(cfg.datasets.galsim_blended_galaxies_coadd.prior)
    decoder = instantiate(cfg.datasets.galsim_blended_galaxies_coadd.decoder)
    background = instantiate(cfg.datasets.galsim_blended_galaxies_coadd.background)

    max_tile_n_sources = 1
    tile_slen = 4
    num_workers = 5
    batch_size = 1000
    n_batches = 1
    num_dithers = 3

    cgb = CoaddGalsimBlends(
        prior=prior,
        decoder=decoder,
        background=background,
        tile_slen=tile_slen,
        max_sources_per_tile=max_tile_n_sources,
        num_workers=num_workers,
        batch_size=batch_size,
        n_batches=n_batches,
        num_dithers=num_dithers
    )
    ds = cgb[0]
    size = cgb.slen + 2 * cgb.bp 

    assert ds["noiseless"].shape == torch.Size([num_dithers, 1, size, size])
    assert ds["background"].shape == torch.Size([num_dithers, 1, size-2, size-2])
    assert ds["images"].shape == torch.Size([1, size-2, size-2])

