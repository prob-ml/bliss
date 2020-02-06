import torch
import torch.nn as nn

from simulated_datasets_lib import _get_mgrid
from psf_transform_lib import get_psf_loss
from psf_transform_lib2 import PowerLawPSF
import utils

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

class BackgroundBias(nn.Module):
    def __init__(self, init_background_params,
                    image_slen =  101):

        super(BackgroundBias, self).__init__()

        assert len(init_background_params.shape) == 2
        self.n_bands = init_background_params.shape[0]

        self.init_background_params = init_background_params.clone()

        self.image_slen = image_slen

        # get grid
        _mgrid = _get_mgrid(image_slen)
        self.mgrid = torch.stack([_mgrid for i in range(self.n_bands)], dim = 0)

        # initial weights
        self.params = nn.Parameter(init_background_params.clone())

    def forward(self):
        return self.params[:, 0][:, None, None] + \
                    self.params[:, 1][:, None, None] * self.mgrid[:, :, :, 0] + \
                    self.params[:, 2][:, None, None] * self.mgrid[:, :, :, 1]

class FluxParams(nn.Module):
    def __init__(self, init_flux_params):
        super(FluxParams, self).__init__()

        self.flux_params = nn.Parameter(init_flux_params.clone())

    def forward(self):
        return self.flux_params

class EstimateModelParams(nn.Module):
    def __init__(self, observed_image, locs, n_stars,
                         init_psf_params,
                         init_background,
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
        assert len(init_background.shape) == 4
        assert observed_image.shape == init_background.shape

        self.observed_image = observed_image
        self.init_background = init_background
        self.slen = observed_image.shape[-1]

        # batchsize
        assert len(n_stars) == locs.shape[0]
        batchsize = locs.shape[0]

        # get n_bands
        assert observed_image.shape[1] == init_psf_params.shape[0]
        assert init_background.shape[1] == init_psf_params.shape[0]
        self.n_bands = init_psf_params.shape[0]

        # get psf
        self.init_psf_params = init_psf_params
        self.power_law_psf = PowerLawPSF(self.init_psf_params)
        self.init_psf = self.power_law_psf.forward().detach()


        self.max_stars = locs.shape[1]
        assert locs.shape[2] == 2

        # boolean for stars being on
        self.is_on_array = utils.get_is_on_from_n_stars(n_stars, self.max_stars)

        # initial flux parameters
        if init_fluxes is None:
            self._init_fluxes(locs)
        else:
            self.init_fluxes = init_fluxes

        self.init_flux_params = torch.log(self.init_fluxes.clamp(min = self.fmin + 1) - self.fmin)
        self.flux_params_class = FluxParams(self.init_flux_params)

        # TODO: pass these as an argument
        self.color_mean = 0.3
        self.color_var = 0.15**2

        # set up background parameters
        self.background_bias = BackgroundBias(image_slen=self.slen,
                        init_background_params=torch.zeros(self.n_bands, 3))

        self.cached_grid = _get_mgrid(observed_image.shape[-1]).to(device)

    def _init_fluxes(self, locs):
        batchsize = locs.shape[0]

        locs_indx = torch.round(locs * (self.slen - 1)).type(torch.long).clamp(max = self.slen - 2,
                                                                            min = 2)

        sky_subtr_image = self.observed_image - self.init_background
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

        self.init_fluxes = self.init_fluxes / self.init_psf.view(self.n_bands, -1).max(1)[0][None, None, :]

    def get_fluxes(self):
        flux_param = self.flux_params_class.forward()
        return torch.exp(flux_param) + self.fmin

    def get_background(self):
        return self.init_background + self.background_bias.forward()

    def get_psf(self):
        return self.power_law_psf.forward()

    def get_loss(self):
        background = self.get_background()
        fluxes = self.get_fluxes()
        psf = self.get_psf()

        recon_mean, loss = get_psf_loss(self.observed_image,
                                            background,
                                            self.locs, fluxes, self.n_stars,
                                            psf,
                                            pad = self.pad,
                                            grid = self.cached_grid)

        return recon_mean, loss

    def _run_optimizer(self, optimizer, tol, max_iter = 20, print_every = False):

        def closure():
            optimizer.zero_grad()
            loss = self.get_loss()[1]
            loss.backward()
            if print_every:
                print(loss)

            return loss

        init_loss = optimizer.step(closure)
        old_loss = init_loss.clone()

        for i in range(1, max_iter):
            loss = optimizer.step(closure)

            if (old_loss - loss) < (init_loss * tol):
                break

        if i == (max_iter - 1):
            print('warning: max iterations reached')
