import torch
from hydra.utils import instantiate


def test_coadd_galsim_blend(get_coadds_config, devices):
    cfg = get_coadds_config({}, devices)
    cgb = instantiate(cfg.datasets.galsim_blends_coadds)
    ds = cgb[0]
    size = cgb.slen + 2 * cgb.bp

    assert ds["noisy"].shape == torch.Size([1, size - 2, size - 2])
    assert ds["background"].shape == torch.Size([1, size - 2, size - 2])
    assert ds["images"].shape == torch.Size([1, size - 2, size - 2])
