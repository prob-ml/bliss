import torch

import torch.nn as nn
from torch.nn.functional import unfold, softmax, pad

import image_utils
from utils import eval_normal_logprob
from simulated_datasets_lib import _get_mgrid, plot_multiple_stars

class PsfLocalTransform(nn.Module):
    def __init__(self, psf,
                    image_slen = 101,
                    kernel_size = 3):

        super(PsfLocalTransform, self).__init__()

        assert len(psf.shape) == 3
        self.n_bands = psf.shape[0]

        assert psf.shape[1] == psf.shape[2]
        self.psf_slen = psf.shape[-1]

        # only implemented for this case atm
        assert image_slen > psf.shape[1]
        assert (image_slen % 2) == 1
        assert (psf.shape[1] % 2) == 1

        self.image_slen = image_slen
        self.kernel_size = kernel_size

        self.psf = psf.unsqueeze(0)
        self.tile_psf()

        # for renormalizing the PSF
        self.normalization = psf.view(self.n_bands, -1).sum(1)

        # initializtion
        init_weight = torch.zeros(self.psf_slen ** 2, self.n_bands,\
                                    kernel_size ** 2)
        init_weight[:, :, 4] = 5
        self.weight = nn.Parameter(init_weight)

    def tile_psf(self):
        psf_unfolded = unfold(self.psf,
                        kernel_size = self.kernel_size,
                        padding = (self.kernel_size - 1) // 2).squeeze(0).transpose(0, 1)


        self.psf_tiled = psf_unfolded.view(psf_unfolded.shape[0], self.n_bands,
                                            self.kernel_size**2)

    def apply_weights(self, w):
        tile_psf_transformed = torch.sum(w * self.psf_tiled, dim = 2).transpose(0, 1)

        return tile_psf_transformed.view(self.n_bands, self.psf_slen,
                                            self.psf_slen)

    def forward(self):
        weights_constrained = torch.nn.functional.softmax(self.weight, dim = 2)

        psf_transformed = self.apply_weights(weights_constrained)

        # pad psf for full image
        l_pad = (self.image_slen - self.psf_slen) // 2
        psf_image = pad(psf_transformed, (l_pad, ) * 4)

        psf_image_normalization = psf_image.view(self.n_bands, -1).sum(1)

        return psf_image * (self.normalization / psf_image_normalization).unsqueeze(-1).unsqueeze(-1)


def get_psf_loss(full_images, full_backgrounds,
                    locs, fluxes, n_stars, psf,
                    pad = 5,
                    grid = None):

    assert len(full_images.shape) == 4
    assert full_images.shape[0] == 1 # for now, one image at a time ...
    n_bands = full_images.shape[1]

    assert full_images.shape == full_backgrounds.shape

    assert len(locs.shape) == 3
    assert len(fluxes.shape) == 3

    assert n_bands == fluxes.shape[2]

    assert len(locs) == len(fluxes)
    assert len(fluxes) == len(n_stars)
    n_samples = len(locs)

    slen = full_images.shape[-1]

    if grid is None:
        grid = _get_mgrid(slen)

    recon_means = \
        plot_multiple_stars(slen, locs, n_stars, fluxes, psf, grid) + \
            full_backgrounds

    _full_image = full_images[:, :, pad:(slen - pad), pad:(slen - pad)].unsqueeze(0)
    _recon_means = recon_means[:, :, pad:(slen - pad), pad:(slen - pad)].clamp(min = 100, max = 2 * _full_image.max())

    # mask = (_recon_means - _full_image).abs() / _full_image < 1.0
    recon_loss = - eval_normal_logprob(_full_image,
                _recon_means,
                torch.log(_recon_means)).view(n_samples, -1).sum(1)

    return recon_means, recon_loss
