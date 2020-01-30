# OK this is silly but lets try this:
import torch
import numpy as np

from torch import nn

import utils

from simulated_datasets_lib import plot_one_star

from torch import optim

class EstimateFluxes(nn.Module):
    def __init__(self, observed_image, locs, n_stars,
                         psf, sky_intensity,
                         alpha = 0.5,
                         pad = 5,
                         init_fluxes = None):

        super(EstimateFluxes, self).__init__()

        self.pad = pad

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
        self.is_on_array = utils.get_is_on_from_n_stars(n_stars, self.max_stars)

        # set star basis
        self.slen = observed_image.shape[-1]
        self.psf = psf
        self.star_basis = \
            plot_one_star(self.slen, locs.view(-1, 2), self.psf,
                          cached_grid = None).view(batchsize,
                                                      self.max_stars,
                                                      self.n_bands,
                                                      self.slen, self.slen) * \
                        self.is_on_array[:, :, None, None, None]

        self.sky_intensity = sky_intensity

        if init_fluxes is None:
            self._init_fluxes(locs)
        else:
            self.init_fluxes = init_fluxes
        self.log_flux = nn.Parameter(torch.log(self.init_fluxes.clone().clamp(min = 1.)))

        self.alpha = alpha
        # TODO: pass these as an argument
        self.color_mean = 0.3
        self.color_var = 0.15**2

    def _init_fluxes(self, locs):
        batchsize = locs.shape[0]

        locs_indx = torch.round(locs * (self.slen - 1)).type(torch.long).clamp(max = self.slen - 2,
                                                                            min = 2)

        sky_subtr_image = self.observed_image - self.sky_intensity[None, :, None, None]
        self.init_fluxes = torch.zeros(batchsize, self.max_stars, self.n_bands)

        for i in range(locs.shape[0]):
            if self.observed_image.shape[0] == 1:
                obs_indx = 0
            else:
                obs_indx = i

            # # take the min over a box of the location
            # init_fluxes_i = torch.zeros(9, self.max_stars, self.n_bands)
            # n = 0
            # for j in [-1, 0, 1]:
            #     for k in [-1, 0, 1]:
            #         init_fluxes_i[n] = sky_subtr_image[obs_indx, :,
            #                             locs_indx[i, :, 0] + j,
            #                             locs_indx[i, :, 1] + k].transpose(0, 1)
            #         n +=1
            #
            # self.init_fluxes[i] = init_fluxes_i.mean(0)

            self.init_fluxes[i] = \
                sky_subtr_image[obs_indx, :,
                    locs_indx[i, :, 0], locs_indx[i, :, 1]].transpose(0, 1)

        self.init_fluxes = self.init_fluxes / self.psf.view(self.n_bands, -1).max(1)[0][None, None, :]

    def forward(self):
        return (torch.exp(self.log_flux[:, :, :, None, None]) * self.star_basis).sum(1) + \
                    self.sky_intensity[None, :, None, None]

    def get_loss(self):
        # log likelihood terms
        recon_mean = self.forward()
        error = 0.5 * ((self.observed_image - recon_mean)**2 / recon_mean) + 0.5 * torch.log(recon_mean)
        neg_loglik = error[:, :, self.pad:(self.slen - self.pad), self.pad:(self.slen - self.pad)].sum()

        # prior terms
        flux_prior = - (self.alpha + 1) * (self.log_flux[:, :, 0] * self.is_on_array).sum()
        if self.n_bands > 1:
            colors = 2.5 * (self.log_flux[:, :, 1:] - self.log_flux[:, :, 0:1]) / np.log(10.)
            color_prior = - 0.5 * (colors - self.color_mean)**2 / self.color_var
            flux_prior += (color_prior * self.is_on_array.unsqueeze(-1)).sum()

        return neg_loglik - flux_prior

    def optimize(self, max_iter = 20):
        optimizer = optim.LBFGS(self.parameters(), max_iter = 20)

        def closure():
            optimizer.zero_grad()
            loss = self.get_loss()
            loss.backward()
            return loss

        optimizer.step(closure)

    def return_fluxes(self):
        return torch.exp(self.log_flux.data)
