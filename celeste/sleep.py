import math
from itertools import permutations
import warnings
import torch
from torch.distributions import Normal
from torch.nn import functional

from . import device


# only function you will ever need to call.
def get_inv_kl_loss(encoder, images, true_locs, true_source_params, use_l2_loss=False):
    """
    NOTE: true_source_params are either log_fluxes or galaxy_params (both are normal unconstrained
    normal variables).

    * images has shape = (batchsize x n_bands x slen x slen)
    * true_locs has shape = (batchsize x max_sources x 2)
    * true_source_params has shape = (batchsize x max_sources x n_source_params)
    """

    # extract image tiles
    # true_tile_locs has shape = (n_ptiles x max_detections x 2)
    # true_tile_n_sources has shape = (n_ptiles)
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
        logprob_bernoulli,
        log_probs_n_sources_per_tile,
    ) = encoder.forward(image_ptiles, n_sources=true_tile_n_sources)

    if use_l2_loss:
        warnings.warn("using l2_loss")
        loc_logvar = torch.zeros(loc_logvar.shape, device=device)
        source_param_logvar = torch.zeros(source_param_logvar.shape, device=device)

    (loss, counter_loss, locs_loss, source_param_loss, perm_indx,) = _get_params_loss(
        loc_mean,
        loc_logvar,
        source_param_mean,
        source_param_logvar,
        log_probs_n_sources_per_tile,
        true_tile_locs,
        true_tile_source_params,
        true_tile_is_on_array.long(),
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
    NOTE: All the quantities except 'true_' are per-tile quantities on first dimension,
    for simplicity not added to names.

    loc_mean shape = (n_ptiles x max_detections x 2)
    source_param_mean shape = (n_ptiles x max_detections x n_source_params)
    true_is_on_array = (n_ptiles x max_detections)
    true_is_on_array = (n_ptiles x max_detections)
    """

    true_n_stars = true_is_on_array.sum(1)
    one_hot_encoding = functional.one_hot(true_n_stars, n_source_log_probs.shape[1])
    counter_loss = _get_categorical_loss(n_source_log_probs, one_hot_encoding)

    locs_log_probs_all = _get_locs_logprob_all_combs(true_locs, loc_mean, loc_logvar)

    source_param_log_probs_all = _get_source_params_logprob_all_combs(
        true_source_params, source_param_mean, source_param_logvar
    )

    locs_loss, source_param_loss, perm_indx = _get_min_perm_loss(
        locs_log_probs_all, source_param_log_probs_all, true_is_on_array
    )

    true_n_stars = true_is_on_array.sum(1)
    one_hot_encoding = functional.one_hot(true_n_stars, n_source_log_probs.shape[1])
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


def _get_transformed_params(true_params, param_mean, param_logvar):
    batchsize = true_params.shape[0]

    # -1 in each view = n_source_params or 2 for locs.
    _true_params = true_params.view(batchsize, true_params.shape[1], 1, -1)
    _source_mean = param_mean.view(batchsize, 1, param_mean.shape[1], -1)
    _source_logvar = param_logvar.view(batchsize, 1, param_logvar.shape[1], -1)

    return _true_params, _source_mean, _source_logvar


def _get_locs_logprob_all_combs(true_locs, loc_mean, loc_logvar):
    # get losses for locations
    # max_detections = loc_mean.shape[1]
    # max_stars = true_locs.shape[1]

    _true_locs, _loc_mean, _loc_logvar = _get_transformed_params(
        true_locs, loc_mean, loc_logvar
    )

    # this will return a large error if star is off
    _true_locs = _true_locs + (_true_locs == 0).float() * 1e16

    # this is batchsize x (max_stars x max_detections)
    # the log prob for each observed location & mean.
    # sum over len(x,y) dimension.
    locs_log_probs_all = (
        Normal(_loc_mean, (torch.exp(_loc_logvar) + 1e-5).sqrt())
        .log_prob(_true_locs)
        .sum(dim=3)
    )
    return locs_log_probs_all


def _get_source_params_logprob_all_combs(
    true_source_params, source_param_mean, source_param_logvar
):
    (
        _true_source_params,
        _source_param_mean,
        _source_param_logvar,
    ) = _get_transformed_params(
        true_source_params, source_param_mean, source_param_logvar
    )

    source_param_log_probs_all = (
        Normal(_source_param_mean, (torch.exp(_source_param_logvar) + 1e-5).sqrt())
        .log_prob(_true_source_params)
        .sum(dim=3)
    )
    return source_param_log_probs_all


def _get_log_probs_all_perms(
    locs_log_probs_all, source_param_log_probs_all, is_on_array
):
    max_detections = source_param_log_probs_all.shape[-1]
    batchsize = source_param_log_probs_all.shape[0]

    locs_loss_all_perm = torch.zeros(
        batchsize, math.factorial(max_detections), device=device
    )
    source_param_loss_all_perm = torch.zeros(
        batchsize, math.factorial(max_detections), device=device
    )

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


# TODO: Can the minus signs here be moved up so that it's a bit clearer?
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
