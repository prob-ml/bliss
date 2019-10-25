import torch
import numpy as np
import math

from torch.distributions import normal

from simulated_datasets_lib import get_is_on_from_n_stars
from hungarian_alg import run_batch_hungarian_alg_parallel

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

from itertools import permutations

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

def get_categorical_loss(log_probs, one_hot_encoding):
    assert torch.all(log_probs <= 0)
    assert log_probs.shape[0] == one_hot_encoding.shape[0]
    assert log_probs.shape[1] == one_hot_encoding.shape[1]

    return torch.sum(
        -log_probs * one_hot_encoding, dim = 1)

def _logit(x, tol = 1e-8):
    return torch.log(x + tol) - torch.log(1 - x + tol)

def eval_normal_logprob(x, mu, log_var):
    return - 0.5 * log_var - 0.5 * (x - mu)**2 / (torch.exp(log_var) + 1e-5) - 0.5 * np.log(2 * np.pi)

def eval_logitnormal_logprob(x, mu, log_var):
    logit_x = _logit(x)
    return eval_normal_logprob(logit_x, mu, log_var)

def eval_lognormal_logprob(x, mu, log_var, tol = 1e-8):
    log_x = torch.log(x + tol)
    return eval_normal_logprob(log_x, mu, log_var)

def _permute_losses_mat(losses_mat, perm):
    batchsize = losses_mat.shape[0]
    max_stars = losses_mat.shape[1]

    assert perm.shape[0] == batchsize
    assert perm.shape[1] == max_stars

    return torch.gather(losses_mat, 2, perm.unsqueeze(2)).squeeze()

def get_locs_logprob_all_combs(true_locs, logit_loc_mean, logit_loc_log_var):

    batchsize = true_locs.shape[0]

    # get losses for locations
    _logit_loc_mean = logit_loc_mean.view(batchsize, 1, logit_loc_mean.shape[1], 2)
    _logit_loc_log_var = logit_loc_log_var.view(batchsize, 1, logit_loc_mean.shape[1], 2)
    _true_locs = true_locs.view(batchsize, true_locs.shape[1], 1, 2)

    # this is batchsize x (max_stars x max_detections)
    # the log prob for each observed location x mean
    locs_log_probs_all = eval_logitnormal_logprob(_true_locs,
                            _logit_loc_mean, _logit_loc_log_var).sum(dim = 3)

    return locs_log_probs_all

def get_fluxes_logprob_all_combs(true_fluxes, log_flux_mean, log_flux_log_var):
    batchsize = true_fluxes.shape[0]

    _log_flux_mean = log_flux_mean.view(batchsize, 1, log_flux_mean.shape[1])
    _log_flux_log_var = log_flux_log_var.view(batchsize, 1, log_flux_mean.shape[1])
    _true_fluxes = true_fluxes.view(batchsize, true_fluxes.shape[1], 1)

    # this is batchsize x (max_stars x max_detections)
    # the log prob for each observed location x mean
    flux_log_probs_all = eval_lognormal_logprob(_true_fluxes,
                                _log_flux_mean, _log_flux_log_var)

    return flux_log_probs_all

# def get_weights_from_n_stars(n_stars):
#     weights = torch.zeros(len(n_stars)).to(device)
#     for i in range(max(n_stars) + 1):
#         weights[n_stars == i] = len(n_stars) / torch.sum(n_stars == i).float()
#
#     return weights / weights.min()
def get_weights_vec(one_hot_encoding, weights):
    # weights_vec= torch.zeros(len(n_stars)).to(device)
    # for i in range(max(n_stars) + 1):
    #     weights_vec[n_stars == i] = weights[i]

    return torch.matmul(one_hot_encoding, weights)

def _get_log_probs_all_perms(locs_log_probs_all, flux_log_probs_all, is_on_array):
    max_detections = flux_log_probs_all.shape[-1]
    batchsize = flux_log_probs_all.shape[0]

    locs_loss_all_perm = torch.zeros(batchsize,
                                        math.factorial(max_detections)).to(device)
    fluxes_loss_all_perm = torch.zeros(batchsize,
                                        math.factorial(max_detections)).to(device)
    i = 0
    for perm in permutations(range(max_detections)):
        locs_loss_all_perm[:, i] = \
            (locs_log_probs_all[:, perm, :].diagonal(dim1 = 1, dim2 = 2) * \
            is_on_array).sum(1)

        fluxes_loss_all_perm[:, i] = \
            (flux_log_probs_all[:, perm].diagonal(dim1 = 1, dim2 = 2) * \
            is_on_array).sum(1)
        i += 1

    return locs_loss_all_perm, fluxes_loss_all_perm

def get_min_perm_loss(locs_log_probs_all, flux_log_probs_all, is_on_array):
    locs_log_probs_all_perm, fluxes_log_probs_all_perm = \
        _get_log_probs_all_perms(locs_log_probs_all, flux_log_probs_all, is_on_array)

    locs_loss, indx = torch.min(-locs_log_probs_all_perm, dim = 1)
    fluxes_loss = -torch.gather(fluxes_log_probs_all_perm, 1, indx.unsqueeze(1)).squeeze()

    return locs_loss, fluxes_loss


def get_params_loss(logit_loc_mean, logit_loc_log_var, \
                        log_flux_mean, log_flux_log_var, log_probs,
                        true_locs, true_fluxes, true_is_on_array,
                        weights):

    max_detections = log_flux_mean.shape[1]

    # this is batchsize x (max_stars x max_detections)
    # the log prob for each observed location x mean
    locs_log_probs_all = \
        get_locs_logprob_all_combs(true_locs,
                                    logit_loc_mean,
                                    logit_loc_log_var)

    flux_log_probs_all = \
        get_fluxes_logprob_all_combs(true_fluxes, \
                                    log_flux_mean, log_flux_log_var)

    locs_loss, fluxes_loss = \
        get_min_perm_loss(locs_log_probs_all, flux_log_probs_all, true_is_on_array)
    # TODO:
    perm = None


    true_n_stars = true_is_on_array.sum(1)
    one_hot_encoding = get_one_hot_encoding_from_int(true_n_stars, log_probs.shape[1])
    counter_loss = get_categorical_loss(log_probs, one_hot_encoding)

    loss_vec = (locs_loss * (locs_loss.detach() < 1e6).float() + fluxes_loss + counter_loss)

    weights_vec = get_weights_vec(one_hot_encoding, weights)
    assert len(weights_vec) == len(loss_vec)
    loss = (loss_vec * weights_vec).mean()
    # loss = loss_vec.mean()

    return loss, counter_loss, locs_loss, fluxes_loss, perm

def get_encoder_loss(star_encoder,
                        images_full,
                        backgrounds_full,
                        true_locs,
                        true_fluxes, use_l2_loss = False):

    # extract image_patches patches
    image_stamps, subimage_locs, subimage_fluxes, true_n_stars, true_is_on_array = \
        star_encoder.get_image_stamps(images_full, true_locs, true_fluxes,
                                        clip_max_stars = True)

    background_stamps = backgrounds_full.mean() # TODO

    # get variational parameters
    logit_loc_mean, logit_loc_log_var, \
        log_flux_mean, log_flux_log_var, log_probs = \
            star_encoder(image_stamps, background_stamps, true_n_stars)

    if use_l2_loss:
        logit_loc_log_var = torch.ones((logit_loc_log_var.shape))
        log_flux_log_var = torch.ones((log_flux_log_var.shape))

    return get_params_loss(logit_loc_mean, logit_loc_log_var, \
                            log_flux_mean, log_flux_log_var, log_probs, \
                            subimage_locs, subimage_fluxes,
                            true_is_on_array.float(),
                            star_encoder.weights)

def eval_star_encoder_loss(star_encoder, train_loader,
                optimizer = None, train = False,
                residual_vae = None):

    avg_loss = 0.0
    avg_counter_loss = 0.0
    avg_locs_loss = 0.0
    avg_fluxes_loss = 0.0

    for _, data in enumerate(train_loader):
        true_fluxes = data['fluxes'].to(device)
        true_locs = data['locs'].to(device)
        images = data['image'].to(device)
        backgrounds = data['background'].to(device)

        if residual_vae is not None:
            # add noise
            residual_vae.eval()
            eta = torch.randn(images.shape[0], residual_vae.latent_dim).to(device)
            residuals = residual_vae.decode(eta)[0] # just taking the mean ...

            images = images * residuals + images

        if train:
            star_encoder.train()
            if optimizer is not None:
                optimizer.zero_grad()
        else:
            star_encoder.eval()

        # evaluate log q
        loss, counter_loss, locs_loss, fluxes_loss = \
            get_encoder_loss(star_encoder, images, backgrounds,
                                true_locs, true_fluxes)[0:4]

        # if(loss > 1000):
        #     print('breaking ... ')
        #     np.savez('./fits/debugging_images', images = images.cpu().numpy(),
                    # locs = true_locs.cpu().numpy(),
                    # fluxes = true_fluxes.cpu().numpy(),
                    # backgrounds = backgrounds.cpu().numpy())
        #     torch.save(star_encoder.state_dict(), './fits/debugging_encoder')

        if train:
            if optimizer is not None:
                loss.backward()
                optimizer.step()

        avg_loss += loss.item() * images.shape[0] / len(train_loader.dataset)
        avg_counter_loss += counter_loss.sum().item() / (len(train_loader.dataset) * star_encoder.n_patches)
        avg_fluxes_loss += fluxes_loss.sum().item() / (len(train_loader.dataset) * star_encoder.n_patches)
        avg_locs_loss += locs_loss.sum().item() / (len(train_loader.dataset) * star_encoder.n_patches)

    return avg_loss, avg_counter_loss, avg_locs_loss, avg_fluxes_loss
