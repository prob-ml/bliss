import torch
import torch.nn as nn
from torch import optim

import numpy as np

from simulated_datasets_lib import _get_mgrid, plot_one_star
from psf_transform_lib import PowerLawPSF
import utils

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def _sample_image(observed_image, sample_every = 10):
    batchsize = observed_image.shape[0]
    n_bands = observed_image.shape[1]
    slen = observed_image.shape[-1]

    samples = torch.zeros(n_bands,
                                int(np.floor(slen / sample_every)),
                                int(np.floor(slen / sample_every)))

    for i in range(samples.shape[1]):
        for j in range(samples.shape[2]):
            x0 = i*sample_every
            x1 = j*sample_every
            samples[:, i, j] = \
                observed_image[:, :,\
                        x0:(x0+sample_every), x1:(x1+sample_every)].reshape(
                                                batchsize,
                                                n_bands, -1).min(2)[0].mean(0)

    return samples

def _fit_plane_to_background(background):
    assert len(background.shape) == 3
    n_bands = background.shape[0]
    slen = background.shape[-1]

    planar_params = np.zeros((n_bands, 3))
    for i in range(n_bands):
        y = background[i].flatten().detach().cpu().numpy()
        grid = _get_mgrid(slen).detach().cpu().numpy()

        x = np.ones((slen**2, 3))
        x[:, 1:] = np.array([grid[:, :, 0].flatten(), grid[:, :, 1].flatten()]).transpose()

        xtx = np.einsum('ki, kj -> ij', x, x)
        xty = np.einsum('ki, k -> i', x, y)

        planar_params[i, :] = np.linalg.solve(xtx, xty)

    return planar_params


class PlanarBackground(nn.Module):
    def __init__(self, init_background_params,
                    image_slen =  101):

        super(PlanarBackground, self).__init__()

        assert len(init_background_params.shape) == 2
        self.n_bands = init_background_params.shape[0]

        self.init_background_params = init_background_params.clone()

        self.image_slen = image_slen

        # get grid
        _mgrid = _get_mgrid(image_slen).to(device)
        self.mgrid = torch.stack([_mgrid for i in range(self.n_bands)], dim = 0)

        # initial weights
        self.params = nn.Parameter(init_background_params.clone())

    def forward(self):
        return self.params[:, 0][:, None, None] + \
                    self.params[:, 1][:, None, None] * self.mgrid[:, :, :, 0] + \
                    self.params[:, 2][:, None, None] * self.mgrid[:, :, :, 1]

class FluxParams(nn.Module):
    def __init__(self, init_fluxes, fmin):
        super(FluxParams, self).__init__()

        self.fmin = fmin
        self.init_flux_params = self._free_flux_params(init_fluxes)
        self.flux_params = nn.Parameter(self.init_flux_params.clone())

    def _free_flux_params(self, fluxes):
        return torch.log(fluxes.clamp(min = self.fmin + 1) - self.fmin)

    def get_fluxes(self):
        return torch.exp(self.flux_params) + self.fmin

class EstimateModelParams(nn.Module):
    def __init__(self, observed_image, locs, n_stars,
                         init_psf_params,
                         init_background_params,
                         init_fluxes = None,
                         fmin = 1e-3,
                         alpha = 0.5,
                         pad = 5):

        super(EstimateModelParams, self).__init__()

        self.pad = pad
        self.alpha = alpha
        self.fmin = fmin
        self.locs = locs
        self.n_stars = n_stars

        # observed image is batchsize (or 1) x n_bands x slen x slen
        assert len(observed_image.shape) == 4

        self.observed_image = observed_image
        self.slen = observed_image.shape[-1]

        # batchsize
        assert len(n_stars) == locs.shape[0]
        self.batchsize = locs.shape[0]

        # get n_bands
        assert observed_image.shape[1] == init_psf_params.shape[0]
        self.n_bands = init_psf_params.shape[0]

        # get psf
        self.init_psf_params = init_psf_params
        self.power_law_psf = PowerLawPSF(self.init_psf_params,
                                            image_slen = self.slen)
        self.init_psf = self.power_law_psf.forward().detach()

        self.max_stars = locs.shape[1]
        assert locs.shape[2] == 2

        # boolean for stars being on
        self.is_on_array = utils.get_is_on_from_n_stars(n_stars, self.max_stars)

        # set up initial background parameters
        if init_background_params is None:
            self._get_init_background()
        else:
            assert init_background_params.shape[0] == self.n_bands
            self.init_background_params = init_background_params

        self.planar_background = PlanarBackground(image_slen=self.slen,
                        init_background_params=self.init_background_params)

        self.init_background = self.planar_background.forward().detach()

        # initial flux parameters
        if init_fluxes is None:
            self._get_init_fluxes()
        else:
            self.init_fluxes = init_fluxes

        self.flux_params_class = FluxParams(self.init_fluxes, self.fmin)

        # TODO: pass these as an argument
        self.color_mean = 0.3
        self.color_var = 0.15**2

        self.cached_grid = _get_mgrid(observed_image.shape[-1]).to(device)
        self._set_star_basis(self.init_psf)

    def _set_star_basis(self, psf):
        self.star_basis = \
            plot_one_star(self.slen, self.locs.view(-1, 2), psf,
                          cached_grid = self.cached_grid).view(self.batchsize,
                                                      self.max_stars,
                                                      self.n_bands,
                                                      self.slen, self.slen) * \
                        self.is_on_array[:, :, None, None, None]

    def _get_init_background(self, sample_every = 25):
        sampled_background = _sample_image(self.observed_image, sample_every)
        self.init_background_params = torch.Tensor(_fit_plane_to_background(sampled_background)).to(device)

    def _get_init_fluxes(self):

        locs_indx = torch.round(self.locs * (self.slen - 1)).type(torch.long).clamp(max = self.slen - 2,
                                                                            min = 2)

        sky_subtr_image = self.observed_image - self.init_background
        self.init_fluxes = torch.zeros(self.batchsize, self.max_stars, self.n_bands).to(device)

        for i in range(self.locs.shape[0]):
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

        self.init_fluxes = self.init_fluxes / self.init_psf.view(self.n_bands, -1).max(1)[0][None, None, :]

    def get_fluxes(self):
        return self.flux_params_class.get_fluxes()

    def get_background(self):
        return self.planar_background.forward().unsqueeze(0)

    def get_psf(self):
        return self.power_law_psf.forward()

    def get_loss(self, use_cached_star_basis = False):
        background = self.get_background()
        fluxes = self.get_fluxes()

        if not use_cached_star_basis:
            psf = self.get_psf()
            self._set_star_basis(psf)
        else:
            self.star_basis = self.star_basis.detach()


        recon_mean = (fluxes[:, :, :, None, None] * self.star_basis).sum(1) + \
                                    background
        recon_mean = recon_mean.clamp(min = 1)

        error = 0.5 * ((self.observed_image - recon_mean)**2 / recon_mean) + 0.5 * torch.log(recon_mean)

        neg_loglik = error[:, :, self.pad:(self.slen - self.pad), self.pad:(self.slen - self.pad)].sum()
        assert ~torch.isnan(neg_loglik)

        # prior terms
        log_flux = torch.log(fluxes)
        flux_prior = - (self.alpha + 1) * (log_flux[:, :, 0] * self.is_on_array).sum()
        if self.n_bands > 1:
            colors = 2.5 * (log_flux[:, :, 1:] - log_flux[:, :, 0:1]) / np.log(10.)
            color_prior = - 0.5 * (colors - self.color_mean)**2 / self.color_var
            flux_prior += (color_prior * self.is_on_array.unsqueeze(-1)).sum()
        assert ~torch.isnan(flux_prior)

        loss = neg_loglik - flux_prior

        return recon_mean, loss

    def _run_optimizer(self, optimizer, tol,
                            use_cached_star_basis = False,
                            max_iter = 20, print_every = False):

        def closure():
            optimizer.zero_grad()
            loss = self.get_loss(use_cached_star_basis)[1]
            loss.backward()

            return loss

        init_loss = optimizer.step(closure)
        old_loss = init_loss.clone()

        for i in range(1, max_iter):
            loss = optimizer.step(closure)

            if print_every:
                print(loss)

            if (old_loss - loss) < (init_loss * tol):
                break

            old_loss = loss

        if max_iter > 1:
            if i == (max_iter - 1):
                print('warning: max iterations reached')

    def optimize_fluxes_background(self, max_iter = 10):
        optimizer1 = optim.LBFGS(list(self.flux_params_class.parameters()) +
                                     list(self.planar_background.parameters()),
                            max_iter = 10,
                            line_search_fn = 'strong_wolfe')

        self._run_optimizer(optimizer1,
                            tol = 1e-3,
                            max_iter = max_iter,
                            use_cached_star_basis = True)

    def run_coordinate_ascent(self, tol = 1e-3,
                                    max_inner_iter = 10,
                                    max_outer_iter = 20):

        old_loss = 1e16
        init_loss = self.get_loss(use_cached_star_basis = True)[1].detach()

        for i in range(max_outer_iter):
            print('\noptimizing fluxes + background. ')
            optimizer1 = optim.LBFGS(list(self.flux_params_class.parameters()) +
                                         list(self.planar_background.parameters()),
                                max_iter = max_inner_iter,
                                line_search_fn = 'strong_wolfe')

            self._run_optimizer(optimizer1, tol = 1e-3, max_iter = 1,
                                use_cached_star_basis = True)

            print('loss: ', self.get_loss(use_cached_star_basis = True)[1].detach())

            print('\noptimizing psf. ')
            psf_optimizer = optim.LBFGS(list(self.power_law_psf.parameters()),
                                max_iter = max_inner_iter,
                                line_search_fn = 'strong_wolfe')

            self._run_optimizer(psf_optimizer, tol = 1e-3, max_iter = 1,
                                        use_cached_star_basis = False)

            loss = self.get_loss(use_cached_star_basis = False)[1].detach()
            print('loss: ', loss)

            if (old_loss - loss) < (tol * init_loss):
                break

            old_loss = loss

        if max_outer_iter > 1:
            if i == (max_outer_iter - 1):
                print('warning: max iterations reached')
