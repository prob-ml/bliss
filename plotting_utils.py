import matplotlib.pyplot as plt

import torch
import numpy as np

from simulated_datasets_lib import plot_multiple_stars

def plot_image(fig, image,
                true_locs = None, estimated_locs = None,
                vmin = None, vmax = None,
                add_colorbar = False,
                global_fig = None,
                diverging_cmap = False):

    slen = image.shape[-1]

    if diverging_cmap:
        im = fig.matshow(image, vmin = vmin, vmax = vmax,
                            cmap=plt.get_cmap('bwr'))
    else:
        im = fig.matshow(image, vmin = vmin, vmax = vmax)

    if not(true_locs is None):
        fig.scatter(x = true_locs[:, 1] * (slen - 1),
                    y = true_locs[:, 0] * (slen - 1),
                    color = 'b')

    if not(estimated_locs is None):
        fig.scatter(x = estimated_locs[:, 1] * (slen - 1),
                    y = estimated_locs[:, 0] * (slen - 1),
                    color = 'r', marker = 'x')

    if add_colorbar:
        assert global_fig is not None
        global_fig.colorbar(im, ax = fig)

def plot_categorical_probs(log_prob_vec, fig):
    n_cat = len(log_prob_vec)
    points = [(i, torch.exp(log_prob_vec[i])) for i in range(n_cat)]

    for pt in points:
        # plot (x,y) pairs.
        # vertical line: 2 x,y pairs: (a,0) and (a,b)
        fig.plot([pt[0],pt[0]], [0,pt[1]], color = 'blue')

    fig.plot(np.arange(n_cat),
             torch.exp(log_prob_vec).detach().numpy(),
             'o', markersize = 5, color = 'blue')

def get_variational_parameters(star_encoder, images,
                               backgrounds, psf,
                               true_n_stars,
                               use_true_n_stars = False):

    if use_true_n_stars:
        n_stars = true_n_stars
    else:
        n_stars = None

    # get parameters
    logit_loc_mean, logit_loc_log_var, \
        log_flux_mean, log_flux_log_var, log_probs = \
            star_encoder(images, backgrounds, n_stars)

    if use_true_n_stars:
        map_n_stars = true_n_stars
    else:
        map_n_stars = torch.argmax(log_probs, dim = 1)


    map_locs = torch.sigmoid(logit_loc_mean).detach()
    map_fluxes = torch.exp(log_flux_mean).detach()

    # get reconstruction
    recon_mean = plot_multiple_stars(map_locs, map_n_stars, map_fluxes, psf) + \
                    backgrounds

    return map_n_stars, map_locs, map_fluxes, \
            logit_loc_mean, logit_loc_log_var, \
                log_flux_mean, log_flux_log_var, \
                    log_probs, recon_mean

def print_results(star_encoder,
                        images,
                        backgrounds,
                        psf,
                        true_locs,
                        true_n_stars,
                        use_true_n_stars = False,
                        residual_clamp = 1e16):

    assert images.shape[0] == backgrounds.shape[0]
    assert images.shape[0] == true_locs.shape[0]
    assert images.shape[0] == len(true_n_stars)

    map_n_stars, map_locs, map_fluxes, \
        logit_loc_mean, logit_loc_log_var, \
            log_flux_mean, log_flux_log_var, \
                log_probs, recon_mean = \
                    get_variational_parameters(star_encoder,
                                            images,
                                            backgrounds,
                                            psf,
                                            true_n_stars,
                                            use_true_n_stars)

    for i in range(images.shape[0]):

        fig, axarr = plt.subplots(1, 4, figsize=(15, 4))

        # observed image
        n_stars_i = true_n_stars[i]
        est_n_stars_i = map_n_stars[i]

        plot_image(axarr[0], images[i, 0, :, :] - backgrounds[i, 0, :, :],
                  true_locs = true_locs[i, 0:int(n_stars_i)],
                  estimated_locs = map_locs[i, 0:int(est_n_stars_i)],
                  add_colorbar = True,
                  global_fig = fig)

        axarr[0].get_xaxis().set_visible(False)
        axarr[0].get_yaxis().set_visible(False)

        axarr[0].set_title('Observed image \n est/true n_stars: {} / {}'.format(est_n_stars_i, int(n_stars_i)))

        # plot posterior samples
        # for k in range(int(est_n_stars_i)):
        #
        #     samples = torch.sigmoid(torch.sqrt(torch.exp(logit_loc_log_var[i, k, :])) * \
        #                   torch.randn((1000, 2)) + logit_loc_mean[i, k, :]).detach()
        #
        #     axarr[1].scatter(x = samples[:, 1] * (images.shape[-1] - 1),
        #                      y = samples[:, 0] * (images.shape[-1] - 1),
        #                      c = 'r', marker = 'x', alpha = 0.05)
        #
        # plot_image(axarr[1], images[i, 0, :, :] - backgrounds[i, 0, :, :],
        #           true_locs = true_locs[i, 0:int(n_stars_i)],
        #           add_colorbar = True,
        #           global_fig = fig)

        plot_image(axarr[1], recon_mean[i, 0, :, :] - backgrounds[i, 0, :, :],
                  estimated_locs = map_locs[i, 0:int(est_n_stars_i)],
                  add_colorbar = True,
                  global_fig = fig)
        axarr[1].get_xaxis().set_visible(False)
        axarr[1].get_yaxis().set_visible(False)

        # plot residual
        residual = images[i, 0, :, :]-recon_mean[i, 0, :, :]
        residual = residual.clamp(min = -residual_clamp, max = residual_clamp)
        vmax = torch.abs(residual).max()

        plot_image(axarr[2], residual,
                  add_colorbar = True,
                  global_fig = fig,
                  diverging_cmap = True,
                  vmin = -vmax, vmax = vmax)

        axarr[2].get_xaxis().set_visible(False)
        axarr[2].get_yaxis().set_visible(False)
        axarr[2].set_title('residuals')

        # plot uncertainty in number of stars
        plot_categorical_probs(log_probs[i], axarr[3])
        axarr[3].set_title('estimated distribution on n_stars')
        axarr[3].plot(np.ones(100) * int(n_stars_i) + 0.05,
                      np.linspace(start = 0,
                                  stop = np.max(np.exp(log_probs.detach().numpy()[i])), num = 100),
                      color = 'red',
                      linestyle = '--')
