import torch
from hydra.utils import instantiate

from case_studies.psf_homogenization.galsim_blends_sg import GalsimBlendsSGwithPSF
from case_studies.psf_homogenization.psf_sampler import PsfSampler


def test_sg_decoder(get_psf_homo_config, devices):
    cfg = get_psf_homo_config({}, devices)
    prior = instantiate(cfg.datasets.galsim_blended_std_psf.prior)
    decoder = instantiate(cfg.datasets.galsim_blended_std_psf.decoder)
    background = instantiate(cfg.datasets.galsim_blended_std_psf.background)
    tile_slen = 4
    max_tile_n_sources = 1
    num_workers = 5
    batch_size = 10
    n_batches = 1
    psf_sampler = PsfSampler()
    std_psf_fwhm = 1.0

    ds_psf = GalsimBlendsSGwithPSF(
        prior,
        decoder,
        background,
        tile_slen,
        max_tile_n_sources,
        num_workers,
        batch_size,
        n_batches,
        psf_sampler,
        std_psf_fwhm,
    )

    for x in ds_psf.train_dataloader():
        assert x["n_sources"].shape == torch.Size([10, 10, 10])
        assert x["images"].shape == torch.Size([10, 1, 88, 88])
        assert x["snr"].shape == torch.Size([10, 10, 10, 1, 1])
        assert x["star_bools"].shape == torch.Size([10, 10, 10, 1, 1])
