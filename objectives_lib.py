import torch

from torch.distributions import normal

from star_datasets_lib import get_is_on_from_n_stars

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def isnan(x):
    return x != x


#############################
# functions to get loss for training the counter
############################
def get_one_hot_encoding_from_int(z, n_classes):
    z = z.long()

    assert len(torch.unique(z)) <= n_classes

    z_one_hot = torch.zeros(len(z), n_classes).to(device)
    z_one_hot.scatter_(1, z.view(-1, 1), 1)
    z_one_hot = z_one_hot.view(len(z), n_classes)

    return z_one_hot

def get_categorical_loss(log_probs, true_n_stars):
    assert torch.all(log_probs <= 0)
    assert log_probs.shape[0] == len(true_n_stars)
    max_detections = log_probs.shape[1]

    return torch.sum(
        -log_probs * \
        get_one_hot_encoding_from_int(true_n_stars,
                                        max_detections), dim = 1)

def eval_star_counter_loss(star_counter, train_loader,
                            optimizer = None, train = True):

    avg_loss = 0.0
    max_detections = torch.Tensor([star_counter.max_detections])

    for _, data in enumerate(train_loader):
        images = data['image'].to(device)
        true_n_stars = data['n_stars'].to(device)

        if train:
            star_counter.train()
            assert optimizer is not None
            optimizer.zero_grad()
        else:
            star_counter.eval()

        # evaluate log q
        log_probs = star_counter(images)
        loss = get_categorical_loss(log_probs, true_n_stars).mean()

        assert not isnan(loss)

        if train:
            loss.backward()
            optimizer.step()

        avg_loss += loss * images.shape[0] / len(train_loader.dataset)

    return avg_loss

#############################
# functions to get loss for training the encoder
############################
def _logit(x, tol = 1e-8):
    return torch.log(x + tol) - torch.log(1 - x + tol)

def _get_normal_logprob(x, mean, logvar):
    norm = normal.Normal(loc = mean, \
                        scale = torch.exp(0.5 * logvar))

    return norm.log_prob(x)

def get_losses_one_detection(true_locs, true_fluxes, true_n_stars,
                            logit_loc_mean_i, logit_loc_logvar_i,
                            log_flux_mean_i, log_flux_logvar_i):

    assert len(logit_loc_mean_i.size()) == 2
    assert len(log_flux_mean_i.size()) == 1

    locs_loss_all = - _get_normal_logprob(_logit(true_locs),
                                logit_loc_mean_i.unsqueeze(1),
                                logit_loc_logvar_i.unsqueeze(1)).sum(dim = 2)

    (locs_loss, perm) = torch.min(locs_loss_all, 1)

    seq_tensor = torch.LongTensor([i for i in range(true_fluxes.shape[0])])
    fluxes_loss = - _get_normal_logprob(torch.log(true_fluxes),
                        log_flux_mean_i.unsqueeze(1),
                        log_flux_logvar_i.unsqueeze(1))[seq_tensor, perm]

    return locs_loss + fluxes_loss, perm


    # max_detections = 1 # logit_loc_mean.size[1]
    #
    # for i in range(max_detections):
    #
    #     # loss for locations
    #     logit_loc_mean_i = logit_loc_mean[:, i, :]
    #     logit_loc_logvar_i = logit_loc_logvar[:, i, :]
    #
    #     locs_loss_all = - _get_normal_logprob(true_locs,
    #                             logit_loc_mean_i.unsqueeze(1),
    #                             logit_loc_logvar_i.unsqueeze(1)).sum(dim = 2)
    #
    #     (locs_loss, perm) = torch.min(locs_loss_all, 1)
    #
    #     # loss for fluxes
    #     log_flux_mean_i = log_flux_mean[:, i]
    #     log_flux_logvar_i = log_flux_logvar[:, i]
    #
    #     seq_tensor = torch.LongTensor([i for i in range(images.shape[0])])
    #     fluxes_loss = - _get_normal_logprob(true_fluxes,
    #                         log_flux_mean_i.unsqueeze(1),
    #                         log_flux_logvar_i.unsqueeze(1))[seq_tensor, perm]
    #
    # return

def get_encoder_loss(star_encoder, images, true_locs,
                        true_fluxes, true_n_stars):

    logit_loc_mean, logit_loc_logvar, \
            log_flux_mean, log_flux_logvar = star_encoder(images, true_n_stars)

    # remove "off" stars
    is_on_array = get_is_on_from_n_stars(true_n_stars,
                                            star_encoder.max_detections)

    true_locs = true_locs * is_on_array.unsqueeze(2) + \
                    1e16 * (1 - is_on_array).unsqueeze(2)

    true_fluxes = true_fluxes * is_on_array + 1e16 * (1 - is_on_array)

    # only one detection at the moment
    i = 0
    logit_loc_mean_i = logit_loc_mean[:, i, :]
    logit_loc_logvar_i = logit_loc_logvar[:, i, :]

    log_flux_mean_i = log_flux_mean[:, i]
    log_flux_logvar_i = log_flux_logvar[:, i]

    loss, perm = get_losses_one_detection(true_locs, true_fluxes, true_n_stars,
                                            logit_loc_mean_i, logit_loc_logvar_i,
                                            log_flux_mean_i, log_flux_logvar_i)
    loss = loss * is_on_array[:, i]
    return loss

def eval_star_encoder_loss(star_encoder, train_loader,
                optimizer = None, train = False):

    avg_loss = 0.0

    for _, data in enumerate(train_loader):
        true_fluxes = data['fluxes'].to(device)
        true_locs = data['locs'].to(device)
        true_n_stars = data['n_stars'].to(device)
        images = data['image'].to(device)

        if train:
            star_encoder.train()
            assert optimizer is not None
            optimizer.zero_grad()
        else:
            star_encoder.eval()

        # evaluate log q
        loss = get_encoder_loss(star_encoder, images, true_locs,
                                true_fluxes, true_n_stars).mean()

        if train:
            loss.backward()
            optimizer.step()

        avg_loss += loss * images.shape[0] / len(train_loader.dataset)

    return avg_loss
