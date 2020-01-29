# OK this is silly but lets try this:
import torch

from torch import nn

import utils

from simulated_datasets_lib import plot_one_star

from torch import optim

class EstimateFluxes(nn.Module):
    def __init__(self, observed_image, locs, n_stars,
                         psf, sky_intensity):

        super(EstimateFluxes, self).__init__()

        # observed image is batchsize (or 1) x n_bands x slen x slen
        assert len(observed_image.shape) == 4
        self.observed_image = observed_image

        # batchsize
        assert len(n_stars) == locs.shape[0]
        batchsize = locs.shape[0]

        # get n_bands
        assert observed_image.shape[1] == psf.shape[0]
        assert len(sky_intensity) == psf.shape[0]
        self.n_bands = psf.shape[0]

        self.max_stars = locs.shape[1]

        assert locs.shape[2] == 2

        # boolean for stars being on
        is_on_array = utils.get_is_on_from_n_stars(n_stars, self.max_stars)

        # set star basis
        self.slen = observed_image.shape[-1]
        self.star_basis = \
            plot_one_star(self.slen, locs.view(-1, 2), psf,
                          cached_grid = None).view(batchsize,
                                                      self.max_stars,
                                                      self.n_bands,
                                                      self.slen, self.slen) * \
                        is_on_array[:, :, None, None, None]

        self.sky_intensity = sky_intensity

        self._init_fluxes(locs)
        self.log_flux = nn.Parameter(torch.log(self.init_fluxes.clone()  + 1e-12) * \
                                        is_on_array.unsqueeze(-1))

    def _init_fluxes(self, locs):
        batchsize = locs.shape[0]

        locs_indx = torch.round(locs * (self.slen - 1)).type(torch.long)

        sky_subtr_image = self.observed_image - self.sky_intensity[None, :, None, None]
        self.init_fluxes = torch.zeros(batchsize, self.max_stars, self.n_bands)
        for i in range(locs.shape[0]):
            if self.observed_image.shape[0] == 1:
                obs_indx = 0
            else:
                obs_indx = i

            self.init_fluxes[i] = \
                sky_subtr_image[obs_indx, :,
                    locs_indx[i, :, 0], locs_indx[i, :, 1]].transpose(0, 1)

    def forward(self):
        return (torch.exp(self.log_flux[:, :, :, None, None]) * self.star_basis).sum(1) + \
                    self.sky_intensity[None, :, None, None]

    def get_loss(self):
        recon_mean = self.forward()
        return ((self.observed_image - recon_mean)**2 / recon_mean).mean()

    def optimize(self):
        optimizer = optim.LBFGS(self.parameters())

        def closure():
            optimizer.zero_grad()
            loss = self.get_loss()
            loss.backward()
            return loss

        optimizer.step(closure)

    def return_fluxes(self):
        return torch.exp(self.log_flux.data)
