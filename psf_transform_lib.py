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

        assert len(psf.shape) == 2
        assert psf.shape[0] == psf.shape[1]

        # only implemented for this case atm
        assert image_slen > psf.shape[1]
        assert (image_slen % 2) == 1
        assert (psf.shape[1] % 2) == 1

        self.image_slen = image_slen
        self.kernel_size = kernel_size

        self.psf = psf.unsqueeze(0).unsqueeze(0)
        self.tile_psf()

        self.normalization = psf.sum()

        self.psf_slen = psf.shape[-1]

        init_weight = torch.zeros(self.psf_slen ** 2, kernel_size ** 2)
        init_weight[:, 4] = 5
        self.weight = nn.Parameter(init_weight)

    def tile_psf(self):
        self.psf_tiled = unfold(self.psf,
                        kernel_size = self.kernel_size,
                        padding = (self.kernel_size - 1) // 2).squeeze().transpose(0, 1)


    def forward(self):
        weights_constrained = torch.nn.functional.softmax(self.weight, dim = 1)

        tile_psf_transformed = torch.sum(weights_constrained * self.psf_tiled, dim = 1)
        psf_transformed = tile_psf_transformed.view(self.psf_slen, self.psf_slen)

        # pad psf for full image
        l_pad = (self.image_slen - self.psf_slen) // 2
        psf_image = pad(psf_transformed, (l_pad, ) * 4)
        return psf_image * self.normalization / psf_image.sum()


def get_psf_loss(full_images, full_backgrounds,
                    locs, fluxes, n_stars, psf,
                    pad = 5,
                    grid = None):

    assert len(full_images.shape) == 4
    assert full_images.shape[0] == 1 # for now, one image at a time ...

    assert full_images.shape == full_backgrounds.shape

    assert len(locs.shape) == 3
    assert len(fluxes.shape) == 2
    assert len(locs) == len(fluxes)
    assert len(fluxes) == len(n_stars)
    n_samples = len(locs)

    slen = full_images.shape[-1]

    if grid is None:
        grid = _get_mgrid(slen)

    n_samples = locs.shape[0]

    recon_loss = 0.0
    for i in range(int(n_samples // 50)):
        indx1 = int(i * 50); print(indx1)
        indx2 = min(int((i + 1) * 50), n_samples)
        recon_means = \
            plot_multiple_stars(slen, locs[indx1:indx2],
                                n_stars[indx1:indx2],
                                fluxes[indx1:indx2], psf, grid) + \
                full_backgrounds

        _full_image = full_images[0, :, pad:(slen - pad), pad:(slen - pad)].unsqueeze(0)
        _recon_means = recon_means[:, :, pad:(slen - pad), pad:(slen - pad)].clamp(min = 100)

        n_samples_i = indx2 - indx1
        recon_loss += - eval_normal_logprob(_full_image,
                                            _recon_means,
                                            torch.log(_recon_means)).view(n_samples_i, -1).sum(1) * \
                        n_samples_i  / n_samples

    return recon_means, recon_loss
