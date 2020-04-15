import math
import time
from itertools import permutations
import numpy as np
import torch

from .utils import const


def isnan(x):
    return x != x


#############################
# functions to get loss for training the counter
############################

def get_categorical_loss(log_probs, one_hot_encoding):
    assert torch.all(log_probs <= 0)
    assert log_probs.shape[0] == one_hot_encoding.shape[0]
    assert log_probs.shape[1] == one_hot_encoding.shape[1]

    return torch.sum(
        -log_probs * one_hot_encoding, dim=1)


def _permute_losses_mat(losses_mat, perm):
    batchsize = losses_mat.shape[0]
    max_stars = losses_mat.shape[1]

    assert perm.shape[0] == batchsize
    assert perm.shape[1] == max_stars

    return torch.gather(losses_mat, 2, perm.unsqueeze(2)).squeeze()


def get_locs_logprob_all_combs(true_locs, loc_mean, loc_log_var):
    batchsize = true_locs.shape[0]

    # get losses for locations
    _loc_mean = loc_mean.view(batchsize, 1, loc_mean.shape[1], 2)
    _loc_log_var = loc_log_var.view(batchsize, 1, loc_mean.shape[1], 2)
    _true_locs = true_locs.view(batchsize, true_locs.shape[1], 1, 2)

    # this will return a large error if star is off
    _true_locs = _true_locs + (_true_locs == 0).float() * 1e16

    # this is batchsize x (max_stars x max_detections)
    # the log prob for each observed location x mean
    locs_log_probs_all = const.eval_normal_logprob(_true_locs,
                                                   _loc_mean, _loc_log_var).sum(dim=3)

    return locs_log_probs_all


def get_source_params_logprob_all_combs(true_source_params, source_param_mean, source_param_log_var,
                                        is_star=True):
    batchsize = true_source_params.shape[0]
    n_source_params = true_source_params.shape[2]

    _log_source_param_mean = log_source_param_mean.view(batchsize, 1, log_source_param_mean.shape[1], n_source_params)
    _log_source_param_log_var = log_source_param_log_var.view(batchsize, 1, log_source_param_mean.shape[1],
                                                              n_source_params)
    _true_source_param = true_source_params.view(batchsize, true_source_params.shape[1], 1, n_source_params)

    # this is batchsize x (max_stars x max_detections)
    # the log prob for each observed location x mean

    # TODO: The reason we need to make this fork is that 'fluxes' are returned from the dataset, but `log_flux_mean` and
    #       `log_flux_logvar` are returned by the `source_encoder.forward` function below.
    if is_star:
        # these are fluxes.
        flux_log_probs_all = const.eval_lognormal_logprob(_true_source_param,
                                                          _log_source_param_mean, _log_source_param_log_var).sum(dim=3)
        return flux_log_probs_all

    else:
        # these are galaxies with latent parameters.
        source_param_log_probs_all = const.eval_normal_logprob(_true_source_param,
                                                               _log_source_param_mean,
                                                               _log_source_param_log_var).sum(dim=3)

        return source_param_log_probs_all


# TODO: other than changing, no need to do anything for fluxes.
def _get_log_probs_all_perms(locs_log_probs_all, flux_log_probs_all, is_on_array):
    max_detections = flux_log_probs_all.shape[-1]
    batchsize = flux_log_probs_all.shape[0]

    locs_loss_all_perm = torch.zeros(batchsize,
                                     math.factorial(max_detections)).cuda()
    fluxes_loss_all_perm = torch.zeros(batchsize,
                                       math.factorial(max_detections)).cuda()
    i = 0
    for perm in permutations(range(max_detections)):
        locs_loss_all_perm[:, i] = \
            (locs_log_probs_all[:, perm, :].diagonal(dim1=1, dim2=2) *
             is_on_array).sum(1)

        fluxes_loss_all_perm[:, i] = \
            (flux_log_probs_all[:, perm].diagonal(dim1=1, dim2=2) *
             is_on_array).sum(1)
        i += 1

    return locs_loss_all_perm, fluxes_loss_all_perm


# TODO: other than changing, no need to do anything for fluxes.
def get_min_perm_loss(locs_log_probs_all, flux_log_probs_all, is_on_array):
    locs_log_probs_all_perm, fluxes_log_probs_all_perm = \
        _get_log_probs_all_perms(locs_log_probs_all, flux_log_probs_all, is_on_array)

    locs_loss, indx = torch.min(-locs_log_probs_all_perm, dim=1)
    fluxes_loss = -torch.gather(fluxes_log_probs_all_perm, 1, indx.unsqueeze(1)).squeeze()

    return locs_loss, fluxes_loss, indx


def get_params_loss(loc_mean, loc_log_var,
                    log_flux_mean, log_flux_log_var, log_probs,
                    true_locs, true_fluxes, true_is_on_array, is_star=True):
    max_detections = log_flux_mean.shape[1]

    # this is batchsize x (max_stars x max_detections)
    # the log prob for each observed location x mean
    locs_log_probs_all = \
        get_locs_logprob_all_combs(true_locs,
                                   loc_mean,
                                   loc_log_var)

    flux_log_probs_all = \
        get_source_params_logprob_all_combs(true_fluxes,
                                            log_flux_mean, log_flux_log_var, is_star=is_star)

    locs_loss, fluxes_loss, perm_indx = \
        get_min_perm_loss(locs_log_probs_all, flux_log_probs_all, true_is_on_array)

    true_n_stars = true_is_on_array.sum(1)
    one_hot_encoding = const.get_one_hot_encoding_from_int(true_n_stars, log_probs.shape[1])
    counter_loss = get_categorical_loss(log_probs, one_hot_encoding)

    loss_vec = (locs_loss * (locs_loss.detach() < 1e6).float() + fluxes_loss + counter_loss)

    loss = loss_vec.mean()

    return loss, counter_loss, locs_loss, fluxes_loss, perm_indx


def get_inv_kl_loss(source_encoder,
                    images,
                    true_locs,
                    true_fluxes, use_l2_loss=False):
    # extract image patches
    image_patches, true_patch_locs, true_patch_source_params, true_patch_n_sources, true_patch_is_on_array = \
        source_encoder.get_image_patches(images, true_locs, true_fluxes,
                                         clip_max_sources=True)

    loc_mean, loc_logvar, log_source_param_mean, log_source_param_logvar, log_probs_n_sources_patch = (
        source_encoder.forward(image_patches, n_sources=true_patch_n_sources)
    )

    if use_l2_loss:
        loc_logvar = torch.zeros(loc_logvar.shape)
        log_source_param_logvar = torch.zeros(log_source_param_logvar.shape)

    loss, counter_loss, locs_loss, source_params_loss, perm_indx = \
        get_params_loss(loc_mean, loc_logvar,
                        log_source_param_mean, log_source_param_logvar, log_probs_n_sources_patch,
                        true_patch_locs, true_patch_source_params,
                        true_patch_is_on_array.float(), is_star=source_encoder.is_star)

    return loss, counter_loss, locs_loss, source_params_loss, perm_indx, log_probs_n_sources_patch


# TODO: Other than changing names, nothing to do for fluxes.
def eval_sleep(source_encoder, dataset, optimizer=None, train=False):
    avg_loss = 0.0
    avg_counter_loss = 0.0
    avg_locs_loss = 0.0
    avg_fluxes_loss = 0.0

    for i, data in enumerate(dataset):
        if i >= len(dataset):  # TODO: is this ok?
            break

        # all are already in cuda.
        true_fluxes = data['source_params']
        true_locs = data['locs']
        images = data['images']

        if train:
            source_encoder.train()
            if optimizer is not None:
                optimizer.zero_grad()
        else:
            source_encoder.eval()

        # evaluate log q
        loss, counter_loss, locs_loss, fluxes_loss = \
            get_inv_kl_loss(source_encoder, images,
                            true_locs, true_fluxes)[0:4]

        if train:
            if optimizer is not None:
                loss.backward()
                optimizer.step()

        avg_loss += loss.item() * images.shape[0] / len(dataset)
        avg_counter_loss += counter_loss.sum().item() / (len(dataset) * source_encoder.n_patches)
        avg_fluxes_loss += fluxes_loss.sum().item() / (len(dataset) * source_encoder.n_patches)
        avg_locs_loss += locs_loss.sum().item() / (len(dataset) * source_encoder.n_patches)

    return avg_loss, avg_counter_loss, avg_locs_loss, avg_fluxes_loss


# TODO: Other than changing names, nothing to do for fluxes.
def run_sleep(source_encoder, dataset, optimizer, n_epochs,
              out_filename, print_every=10):
    train_losses = np.zeros((4, n_epochs))

    for epoch in range(n_epochs):
        t0 = time.time()

        avg_loss, counter_loss, locs_loss, fluxes_loss = \
            eval_sleep(source_encoder, dataset, optimizer, train=True)

        elapsed = time.time() - t0
        print(
            f'{epoch} loss: {avg_loss:.4f}; counter loss: {counter_loss:.4f}; locs loss: {locs_loss:.4f}; fluxes loss: '
            f'{fluxes_loss:.4f} \t [{elapsed:.1f} seconds]'
        )

        train_losses[:, epoch] = np.array([avg_loss, counter_loss, locs_loss, fluxes_loss])
        np.savetxt(f"{out_filename}-train_losses", train_losses)

        if (epoch % print_every) == 0:
            test_loss, test_counter_loss, test_locs_loss, test_fluxes_loss = \
                eval_sleep(source_encoder, dataset, train=False)

            print(
                f'**** test loss: {test_loss:.3f}; counter loss: {test_counter_loss:.3f}; '
                f'locs loss: {test_locs_loss:.3f}; fluxes loss: {test_fluxes_loss:.3f} ****'
            )

            print("writing the encoder parameters to " + out_filename.as_posix())
            torch.save(source_encoder.state_dict(), out_filename)
