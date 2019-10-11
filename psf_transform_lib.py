import torch

import torch.nn as nn
from torch.nn.functional import unfold, softmax, pad

import image_utils
from kl_objective_lib import eval_normal_logprob

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

        self.psf_slen = psf.shape[-1]

        self.weight = nn.Parameter(
            torch.randn(self.psf_slen ** 2, kernel_size ** 2)
        )

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
        return psf_image


def get_psf_transform_loss(full_images, full_backgrounds,
                            subimage_locs,
                            subimage_fluxes,
                            tile_coords,
                            subimage_slen,
                            edge_padding,
                            simulator,
                            psf_transform):

    locs_full_image, fluxes_full_image, _ = \
        image_utils.get_full_params_from_patch_params(subimage_locs, subimage_fluxes,
                                                tile_coords,
                                                full_images.shape[-1],
                                                subimage_slen,
                                                edge_padding,
                                                full_images.shape[0])

    simulator.psf = psf_transform.forward()
    recon_means = simulator.draw_image_from_params(locs = locs_full_image,
                                                  fluxes = fluxes_full_image,
                                                  n_stars = torch.sum(fluxes_full_image > 0, dim = 1),
                                                  add_noise = False)


    recon_means = recon_means - simulator.sky_intensity + full_backgrounds
    recon_loss = - eval_normal_logprob(full_images, recon_means, torch.log(recon_means)).view(full_images.shape[0], -1).sum(1)

    return recon_means, recon_loss
