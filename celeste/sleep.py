import math
from itertools import permutations
import torch

from . import utils


# only function you will ever need to call.
def get_inv_kl_loss(encoder, images, true_locs, true_source_params):
    """
    NOTE: true_source_params are either log_fluxes or galaxy_params (both are normal unconstrained
    normal variables).
    """
    # extract image tiles
    (
        image_ptiles,
        true_tile_locs,
        true_tile_source_params,
        true_tile_n_sources,
        true_tile_is_on_array,
    ) = encoder.get_image_ptiles(
        images, true_locs, true_source_params, clip_max_sources=True
    )

    (
        loc_mean,
        loc_logvar,
        source_param_mean,
        source_param_logvar,
        log_probs_n_sources_per_tile,
    ) = encoder.forward(image_ptiles, n_sources=true_tile_n_sources)

    (loss, counter_loss, locs_loss, source_param_loss, perm_indx,) = _get_params_loss(
        loc_mean,
        loc_logvar,
        source_param_mean,
        source_param_logvar,
        log_probs_n_sources_per_tile,
        true_tile_locs,
        true_tile_source_params,
        true_tile_is_on_array.float(),
    )

    return (
        loss,
        counter_loss,
        locs_loss,
        source_param_loss,
        perm_indx,
        log_probs_n_sources_per_tile,
    )


def _get_params_loss(
    loc_mean,
    loc_logvar,
    source_param_mean,
    source_param_logvar,
    n_source_log_probs,
    true_locs,
    true_source_params,
    true_is_on_array,
):
    """
    NOTE: All the quantities are per-tile quantities on first dimension, for simplicity not added
    to names.
    """
    # this is batchsize x (max_stars x max_detections)
    # max_detections = log_flux_mean.shape[1]
    # the log prob for each observed location x mean
    locs_log_probs_all = _get_locs_logprob_all_combs(true_locs, loc_mean, loc_logvar)

    source_param_log_probs_all = _get_source_params_logprob_all_combs(
        true_source_params, source_param_mean, source_param_logvar
    )

    locs_loss, source_param_loss, perm_indx = _get_min_perm_loss(
        locs_log_probs_all, source_param_log_probs_all, true_is_on_array
    )

    true_n_stars = true_is_on_array.sum(1)
    one_hot_encoding = utils.get_one_hot_encoding_from_int(
        true_n_stars, n_source_log_probs.shape[1]
    )
    counter_loss = _get_categorical_loss(n_source_log_probs, one_hot_encoding)

    loss_vec = (
        locs_loss * (locs_loss.detach() < 1e6).float()
        + source_param_loss
        + counter_loss
    )

    loss = loss_vec.mean()

    return loss, counter_loss, locs_loss, source_param_loss, perm_indx


def _get_categorical_loss(n_source_log_probs, one_hot_encoding):
    assert torch.all(n_source_log_probs <= 0)
    assert n_source_log_probs.shape[0] == one_hot_encoding.shape[0]
    assert n_source_log_probs.shape[1] == one_hot_encoding.shape[1]

    return torch.sum(-n_source_log_probs * one_hot_encoding, dim=1)


def _get_locs_logprob_all_combs(true_locs, loc_mean, loc_log_var):
    batchsize = true_locs.shape[0]

    # get losses for locations
    _loc_mean = loc_mean.view(batchsize, 1, loc_mean.shape[1], 2)
    _loc_log_var = loc_log_var.view(batchsize, 1, loc_mean.shape[1], 2)
    _true_locs = true_locs.view(batchsize, true_locs.shape[1], 1, 2)

    # this will return a large error if star is off
    _true_locs = _true_locs + (_true_locs == 0).float() * 1e16

    # this is batchsize x (max_stars x max_detections)
    # the log prob for each observed location x mean
    locs_log_probs_all = utils.eval_normal_logprob(
        _true_locs, _loc_mean, _loc_log_var
    ).sum(dim=3)

    return locs_log_probs_all


def _get_source_params_logprob_all_combs(
    true_source_params, source_param_mean, source_param_logvar
):
    (
        _true_source_params,
        _source_param_mean,
        _source_param_logvar,
    ) = _get_transformed_source_params(
        true_source_params, source_param_mean, source_param_logvar
    )
    source_param_log_probs_all = utils.eval_normal_logprob(
        _true_source_params, _source_param_mean, _source_param_logvar
    ).sum(dim=3)
    return source_param_log_probs_all


def _permute_losses_mat(losses_mat, perm):
    batchsize = losses_mat.shape[0]
    max_stars = losses_mat.shape[1]

    assert perm.shape[0] == batchsize
    assert perm.shape[1] == max_stars

    return torch.gather(losses_mat, 2, perm.unsqueeze(2)).squeeze()


def _get_log_probs_all_perms(
    locs_log_probs_all, source_param_log_probs_all, is_on_array
):
    max_detections = source_param_log_probs_all.shape[-1]
    batchsize = source_param_log_probs_all.shape[0]

    locs_loss_all_perm = torch.zeros(batchsize, math.factorial(max_detections)).cuda()
    source_param_loss_all_perm = torch.zeros(
        batchsize, math.factorial(max_detections)
    ).cuda()
    i = 0
    for perm in permutations(range(max_detections)):
        locs_loss_all_perm[:, i] = (
            locs_log_probs_all[:, perm, :].diagonal(dim1=1, dim2=2) * is_on_array
        ).sum(1)

        source_param_loss_all_perm[:, i] = (
            source_param_log_probs_all[:, perm].diagonal(dim1=1, dim2=2) * is_on_array
        ).sum(1)
        i += 1

    return locs_loss_all_perm, source_param_loss_all_perm


def _get_min_perm_loss(locs_log_probs_all, source_params_log_probs_all, is_on_array):
    (
        locs_log_probs_all_perm,
        source_params_log_probs_all_perm,
    ) = _get_log_probs_all_perms(
        locs_log_probs_all, source_params_log_probs_all, is_on_array
    )

    locs_loss, indx = torch.min(-locs_log_probs_all_perm, dim=1)
    source_params_loss = -torch.gather(
        source_params_log_probs_all_perm, 1, indx.unsqueeze(1)
    ).squeeze()

    return locs_loss, source_params_loss, indx


def _get_transformed_source_params(
    true_source_params, source_param_mean, source_param_logvar
):
    n_tiles = true_source_params.shape[0]

    # -1 in each view = n_source_params
    _true_source_params = true_source_params.view(
        n_tiles, true_source_params.shape[1], 1, -1
    )
    _source_param_mean = source_param_mean.view(
        n_tiles, 1, source_param_mean.shape[1], -1
    )
    _source_param_logvar = source_param_logvar.view(
        n_tiles, 1, source_param_mean.shape[1], -1
    )

    return _true_source_params, _source_param_mean, _source_param_logvar
