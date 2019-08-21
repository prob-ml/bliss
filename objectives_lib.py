import torch

from torch.distributions import normal

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def logit(x):
    return torch.log(x) - torch.log(1 - x)

def get_invKL_loss(star_rnn, images, true_fluxes, true_locs, true_n_stars):
    # loss for the first detection only right now

    h_i = torch.zeros(images.shape[0], star_rnn.hidden_length).to(device)

    # forward
    logit_locs_mean, logit_locs_logvar, \
        log_flux_mean, log_flux_logvar = \
            star_rnn.forward_once(images, h_i)
    # get loss
    logit_locs_q = normal.Normal(loc = logit_locs_mean.unsqueeze(1), \
                                scale = torch.exp(0.5 * logit_locs_logvar).unsqueeze(1))
    log_flux_q = normal.Normal(loc = log_flux_mean.unsqueeze(1), \
                                scale = torch.exp(0.5 * log_flux_logvar).unsqueeze(1))

    # locs loss
    locs_loss_all = - logit_locs_q.log_prob(logit(true_locs)).sum(dim = 2)
    (locs_loss, perm) = torch.min(locs_loss_all, 1)

    seq_tensor = torch.LongTensor([i for i in range(images.shape[0])])
    fluxes_loss = - log_flux_q.log_prob(torch.log(true_fluxes))[seq_tensor, perm]

    return locs_loss + fluxes_loss, perm
