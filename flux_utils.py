# OK this is silly but lets try this:
import torch
import numpy as np

from torch import nn

import utils

from simulated_datasets_lib import plot_one_star

from torch import optim

class EstimateFluxes(nn.Module):
    def __init__(self, observed_image, locs, n_stars,
                         psf, background,
                         fmin,
                         alpha = 0.5,
                         pad = 5,
                         init_fluxes = None):

        super(EstimateFluxes, self).__init__()

        self.pad = pad
        self.fmin = fmin

        # observed image is batchsize (or 1) x n_bands x slen x slen
        assert len(observed_image.shape) == 4
        assert len(background.shape) == 4
        self.observed_image = observed_image
        self.background = background

        # batchsize
        assert len(n_stars) == locs.shape[0]
        batchsize = locs.shape[0]

        # get n_bands
        assert observed_image.shape[1] == psf.shape[0]
        assert background.shape[1] == psf.shape[0]
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

        if init_fluxes is None:
            self._init_fluxes(locs)
        else:
            self.init_fluxes = init_fluxes

        self.init_param = torch.log(self.init_fluxes.clamp(min = self.fmin + 1) - self.fmin)
        self.param = nn.Parameter(self.init_param.clone())

        self.alpha = alpha
        # TODO: pass these as an argument
        self.color_mean = 0.3
        self.color_var = 0.15**2

        self.init_loss = self.get_loss()

    def _init_fluxes(self, locs):
        batchsize = locs.shape[0]

        locs_indx = torch.round(locs * (self.slen - 1)).type(torch.long).clamp(max = self.slen - 2,
                                                                            min = 2)

        sky_subtr_image = self.observed_image - self.background
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
        fluxes = torch.exp(self.param[:, :, :, None, None]) + self.fmin
        recon_mean = (fluxes * self.star_basis).sum(1) + \
                    self.background
        return recon_mean.clamp(min = 1e-6)

    def get_loss(self):
        # log likelihood terms
        recon_mean = self.forward()
        error = 0.5 * ((self.observed_image - recon_mean)**2 / recon_mean) + 0.5 * torch.log(recon_mean)
        assert (~torch.isnan(error)).all()

        neg_loglik = error[:, :, self.pad:(self.slen - self.pad), self.pad:(self.slen - self.pad)].sum()

        # prior terms
        log_flux = self.param + np.log(self.fmin)
        flux_prior = - (self.alpha + 1) * (log_flux[:, :, 0] * self.is_on_array).sum()
        if self.n_bands > 1:
            colors = 2.5 * (log_flux[:, :, 1:] - log_flux[:, :, 0:1]) / np.log(10.)
            color_prior = - 0.5 * (colors - self.color_mean)**2 / self.color_var
            flux_prior += (color_prior * self.is_on_array.unsqueeze(-1)).sum()

        assert ~torch.isnan(flux_prior)

        loss = neg_loglik - flux_prior

        return loss

    def optimize(self,
                max_outer_iter = 10,
                max_inner_iter = 20,
                tol = 1e-3,
                print_every = False):

        optimizer = optim.LBFGS(self.parameters(),
                            max_iter = max_inner_iter,
                            line_search_fn = 'strong_wolfe')

        def closure():
            optimizer.zero_grad()
            loss = self.get_loss()
            loss.backward()
            return loss

        old_loss = 1e16
        for i in range(max_outer_iter):
            optimizer.step(closure)

            loss = self.get_loss()

            if print_every:
                print(loss)

            diff = (loss - old_loss).abs()
            if diff < (tol * self.init_loss):
                break

            old_loss = loss

    def return_fluxes(self):
        return torch.exp(self.param.data) + self.fmin
