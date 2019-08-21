import torch

from torch.distributions import normal

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def get_invKL_loss(star_rnn, images, true_fluxes, true_locs, true_n_stars):
    # loss for the first detection only right now

    # forward
    logit_locs_mean, logit_locs_logvar, \
        log_flux_mean, log_flux_logvar = \
            star_rnn.forward_once(images, \
                                    h_i = torch.zeros(images.shape[0], 180).to(device))

    # get loss
    logit_locs_q = normal.Normal(loc = logit_locs_mean, \
                                scale = torch.exp(0.5 * logit_locs_logvar))
    log_flux_q = normal.Normal(loc = log_flux_mean, \
                                scale = torch.exp(0.5 * log_flux_logvar))

    loss = -logit_locs_q.log_prob(true_locs[:, 0, :]).sum(dim = 1) - \
                log_flux_q.log_prob(true_fluxes[:, 0])

    return loss
