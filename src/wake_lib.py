import torch
import torch.nn as nn
from torch import optim

import numpy as np

from GalaxyModel.src import _get_mgrid, plot_multiple_stars
from GalaxyModel.src import PowerLawPSF

import time

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def _sample_image(observed_image, sample_every=10):
    batchsize = observed_image.shape[0]
    n_bands = observed_image.shape[1]
    slen = observed_image.shape[-1]

    samples = torch.zeros(n_bands,
                          int(np.floor(slen / sample_every)),
                          int(np.floor(slen / sample_every)))

    for i in range(samples.shape[1]):
        for j in range(samples.shape[2]):
            x0 = i * sample_every
            x1 = j * sample_every
            samples[:, i, j] = \
                observed_image[:, :, \
                x0:(x0 + sample_every), x1:(x1 + sample_every)].reshape(
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

        x = np.ones((slen ** 2, 3))
        x[:, 1:] = np.array([grid[:, :, 0].flatten(), grid[:, :, 1].flatten()]).transpose()

        xtx = np.einsum('ki, kj -> ij', x, x)
        xty = np.einsum('ki, k -> i', x, y)

        planar_params[i, :] = np.linalg.solve(xtx, xty)

    return planar_params


class PlanarBackground(nn.Module):
    def __init__(self, init_background_params,
                 image_slen=101):
        super(PlanarBackground, self).__init__()

        assert len(init_background_params.shape) == 2
        self.n_bands = init_background_params.shape[0]

        self.init_background_params = init_background_params.clone()

        self.image_slen = image_slen

        # get grid
        _mgrid = _get_mgrid(image_slen).to(device)
        self.mgrid = torch.stack([_mgrid for i in range(self.n_bands)], dim=0)

        # initial weights
        self.params = nn.Parameter(init_background_params.clone())

    def forward(self):
        return self.params[:, 0][:, None, None] + \
               self.params[:, 1][:, None, None] * self.mgrid[:, :, :, 0] + \
               self.params[:, 2][:, None, None] * self.mgrid[:, :, :, 1]


class ModelParams(nn.Module):
    def __init__(self, observed_image,
                 init_psf_params,
                 init_background_params,
                 pad=5):

        super(ModelParams, self).__init__()

        self.pad = pad

        # observed image is batchsize (or 1) x n_bands x slen x slen
        assert len(observed_image.shape) == 4

        self.observed_image = observed_image
        self.slen = observed_image.shape[-1]

        # get n_bands
        assert observed_image.shape[1] == init_psf_params.shape[0]
        self.n_bands = init_psf_params.shape[0]

        # get psf
        self.init_psf_params = init_psf_params
        # if image slen is even, add one. psf dimension must be odd
        psf_slen = self.slen + ((self.slen % 2) == 0) * 1
        self.power_law_psf = PowerLawPSF(self.init_psf_params,
                                         image_slen=psf_slen)
        self.init_psf = self.power_law_psf.forward().detach().clone()
        self.psf = self.power_law_psf.forward()

        # set up initial background parameters
        if init_background_params is None:
            self._get_init_background()
        else:
            assert init_background_params.shape[0] == self.n_bands
            self.init_background_params = init_background_params
            self.planar_background = PlanarBackground(image_slen=self.slen,
                                                      init_background_params=self.init_background_params)

        self.init_background = self.planar_background.forward().detach()

        self.cached_grid = _get_mgrid(observed_image.shape[-1]).to(device)

    def _plot_stars(self, locs, fluxes, n_stars, psf):
        self.stars = plot_multiple_stars(self.slen, locs, n_stars,
                                         fluxes, psf, self.cached_grid)

    def _get_init_background(self, sample_every=25):
        sampled_background = _sample_image(self.observed_image, sample_every)
        self.init_background_params = torch.Tensor(_fit_plane_to_background(sampled_background)).to(device)
        self.planar_background = PlanarBackground(image_slen=self.slen,
                                                  init_background_params=self.init_background_params)

    def get_background(self):
        return self.planar_background.forward().unsqueeze(0)

    def get_psf(self):
        return self.power_law_psf.forward()

    def get_loss(self, use_cached_stars=False,
                 locs=None, fluxes=None, n_stars=None):
        background = self.get_background()

        if not use_cached_stars:
            assert locs is not None
            assert fluxes is not None
            assert n_stars is not None

            psf = self.get_psf()
            self._plot_stars(locs, fluxes, n_stars, psf)
        else:
            assert hasattr(self, 'stars')
            self.stars = self.stars.detach()

        recon_mean = self.stars + background

        error = 0.5 * ((self.observed_image - recon_mean) ** 2 / recon_mean) + 0.5 * torch.log(recon_mean)

        loss = error[:, :, self.pad:(self.slen - self.pad),
               self.pad:(self.slen - self.pad)].reshape(error.shape[0], -1).sum(1)

        return recon_mean, loss


def get_wake_loss(image, star_encoder, model_params, n_samples, run_map=False):
    locs_sampled, fluxes_sampled, n_stars_sampled = \
        star_encoder.sample_star_encoder(image,
                                         return_map_n_stars=run_map,
                                         return_map_star_params=run_map,
                                         n_samples=n_samples)[0:3]

    loss = model_params.get_loss(locs=locs_sampled.detach(),
                                 fluxes=fluxes_sampled.detach(),
                                 n_stars=n_stars_sampled.detach())[1].mean()

    return loss


def run_wake(image, star_encoder, init_psf_params,
             init_background_params,
             n_samples,
             out_filename,
             n_epochs=100,
             lr=1e-3,
             print_every=20,
             run_map=False):
    model_params = ModelParams(image,
                               init_psf_params,
                               init_background_params)

    avg_loss = 0.0
    counter = 0
    t0 = time.time()
    test_losses = []

    # optimizer = optim.Adam([{'params': model_params.power_law_psf.parameters(),
    #                         'lr': lr},
    #                         {'params': model_params.planar_background.parameters(),
    #                         'lr': lr}])
    optimizer = optim.Adam([{'params': model_params.power_law_psf.parameters(),
                             'lr': lr}])

    if run_map:
        n_samples = 1

    for epoch in range(1, n_epochs + 1):

        optimizer.zero_grad()

        loss = get_wake_loss(image, star_encoder, model_params,
                             n_samples, run_map)

        loss.backward()
        optimizer.step()

        # avg_loss += loss.detach()
        # counter += 1

        if ((epoch % print_every) == 0) or (epoch == n_epochs):
            eval_loss = get_wake_loss(image, star_encoder, model_params,
                                      n_samples=1, run_map=True).detach()

            elapsed = time.time() - t0
            print('[{}] loss: {:0.4f} \t[{:.1f} seconds]'.format( \
                epoch, eval_loss, elapsed))

            test_losses.append(eval_loss)
            np.savetxt(out_filename + '-wake_losses', test_losses)

            # reset
            avg_loss = 0.0
            counter = 0
            t0 = time.time()

        np.save(out_filename + '-powerlaw_psf_params',
                list(model_params.power_law_psf.parameters())[0].data.cpu().numpy())
        np.save(out_filename + '-planarback_params',
                list(model_params.planar_background.parameters())[0].data.cpu().numpy())

    map_loss = get_wake_loss(image, star_encoder, model_params,
                             n_samples=1, run_map=True)

    return model_params, map_loss
