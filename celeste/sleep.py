import math
from itertools import permutations
import warnings

import torch
from torch.distributions import Normal
from torch.nn import functional

from . import device


def get_inv_kl_loss(
    encoder,
    images,
    true_locs,
    true_galaxy_params,
    true_log_fluxes,
    true_galaxy_bool,
    use_l2_loss=False,
):

    # extract image tiles
    # true_tile_locs has shape = (n_ptiles x max_detections x 2)
    # true_tile_n_sources has shape = (n_ptiles)
    (
        image_ptiles,
        true_tile_locs,
        true_tile_galaxy_params,
        true_tile_log_fluxes,
        true_tile_n_sources,
        true_tile_galaxy_bool,
        true_tile_is_on_array,
    ) = encoder.get_image_ptiles(
        images, true_locs, true_galaxy_params, true_log_fluxes, true_galaxy_bool,
    )

    (
        loc_mean,
        loc_logvar,
        galaxy_param_mean,
        galaxy_param_logvar,
        log_flux_mean,
        log_flux_logvar,
        prob_galaxy,
        n_source_log_probs,
    ) = encoder.forward(image_ptiles, n_sources=true_tile_n_sources)

    if use_l2_loss:
        warnings.warn("using l2_loss")
        loc_logvar = torch.zeros_like(loc_logvar)
        galaxy_param_logvar = torch.zeros_like(galaxy_param_logvar)
        log_flux_logvar = torch.zeros_like(log_flux_logvar)

    (
        loss,
        counter_loss,
        galaxy_bool_loss,
        locs_loss,
        galaxy_params_loss,
        star_params_loss,
        perm_indx,
    ) = _get_params_loss(
        loc_mean,
        loc_logvar,
        galaxy_param_mean,
        galaxy_param_logvar,
        log_flux_mean,
        log_flux_logvar,
        prob_galaxy,
        n_source_log_probs,
        true_tile_locs,
        true_tile_galaxy_params,
        true_tile_log_fluxes,
        true_tile_galaxy_bool,
        true_tile_is_on_array.long(),
    )

    return (
        loss,
        counter_loss,
        locs_loss,
        galaxy_params_loss,
        star_params_loss,
    )


def _get_params_loss(
    loc_mean,
    loc_logvar,
    galaxy_params_mean,
    galaxy_params_logvar,
    log_flux_mean,
    log_flux_logvar,
    prob_galaxy,
    n_source_log_probs,
    true_locs,
    true_galaxy_params,
    true_log_fluxes,
    true_galaxy_bool,
    true_is_on_array,
):
    """
    NOTE: All the quantities are per-tile quantities on first dimension,
    for simplicity not added to names.

    loc_mean shape = (n_ptiles x max_detections x 2)
    log_flux_mean shape = (n_ptiles x max_detections x n_bands)
    galaxy_params_mean shape = (n_ptiles x max_detections x n_galaxy_params)

    the *_logvar inputs should the same shape as their respective means
    the true_* inputs, except for true_is_on_array,
    should have same shape as their respective means, e.g.
    true_locs should have the same shape as loc_mean

    In true_locs, the off sources must have parameter value = 0

    true_is_on_array shape = (n_ptiles x max_detections)
        Indicates if sources is on (1) or off (0)

    true_galaxy_bool shape = (n_ptiles x max_detections)
        indicating whether each source is a galaxy (1) or star (0)

    prob_galaxy shape = (n_ptiles x max_detections)
        are probabilities for each source to be a galaxy

    n_source_log_probs shape = (ptiles x (max_detections + 1))
        are log-probabilities for the number of sources (0, 1, ..., max_detections)

    """

    # the loss for estimating the true number of sources
    true_n_sources = true_is_on_array.sum(1)
    one_hot_encoding = functional.one_hot(true_n_sources, n_source_log_probs.shape[1])
    counter_loss = _get_categorical_loss(n_source_log_probs, one_hot_encoding)

    # the following three functions computes the log-probability of parameters when
    # each estimated source i is matched with true source j for
    # i, j in {1, ..., max_detections}
    # *_log_probs_all have shape n_ptiles x max_detections x max_detections

    # large error if source is off
    true_locs = true_locs + (true_locs == 0).float() * 1e16
    locs_log_probs_all = _get_params_logprob_all_combs(true_locs, loc_mean, loc_logvar)

    galaxy_params_log_probs_all = _get_params_logprob_all_combs(
        true_galaxy_params, galaxy_params_mean, galaxy_params_logvar
    )

    star_params_log_probs_all = _get_params_logprob_all_combs(
        true_log_fluxes, log_flux_mean, log_flux_logvar
    )

    # inside _get_min_perm_loss is where the matching happens:
    # we construct a bijective map from each estimated star to each true star
    locs_loss, star_params_loss, galaxy_params_loss, perm_indx = _get_min_perm_loss(
        locs_log_probs_all,
        star_params_log_probs_all,
        galaxy_params_log_probs_all,
        true_galaxy_bool,
        true_is_on_array,
    )

    # loss for detecting galaxies
    galaxy_bool_loss = true_galaxy_bool * torch.log(prob_galaxy)
    galaxy_bool_loss += (1 - true_galaxy_bool) * torch.log1p(prob_galaxy)
    galaxy_bool_loss = (galaxy_bool_loss * true_is_on_array).sum(1)

    loss_vec = (
        locs_loss * (locs_loss.detach() < 1e6).float()
        + star_params_loss
        + galaxy_params_loss
        + counter_loss
        + galaxy_bool_loss
    )

    loss = loss_vec.mean()

    return (
        loss,
        counter_loss,
        galaxy_bool_loss,
        locs_loss,
        galaxy_params_loss,
        star_params_loss,
        perm_indx,
    )


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


def _get_params_logprob_all_combs(true_params, param_mean, param_logvar):
    (_true_params, _param_mean, _param_logvar,) = _get_transformed_params(
        true_params, param_mean, param_logvar
    )

    param_log_probs_all = (
        Normal(_param_mean, (torch.exp(_param_logvar) + 1e-5).sqrt())
        .log_prob(_true_params)
        .sum(dim=3)
    )
    return param_log_probs_all


def _get_log_probs_all_perms(
    locs_log_probs_all,
    star_params_log_probs_all,
    galaxy_params_log_probs_all,
    is_on_array,
    true_galaxy_bool,
):

    # get log-probability under every possible matching of estimated star to true star
    max_detections = galaxy_params_log_probs_all.shape[-1]
    batchsize = galaxy_params_log_probs_all.shape[0]

    locs_loss_all_perm = torch.zeros(
        batchsize, math.factorial(max_detections), device=device
    )
    star_params_loss_all_perm = torch.zeros(
        batchsize, math.factorial(max_detections), device=device
    )
    galaxy_params_loss_all_perm = torch.zeros(
        batchsize, math.factorial(max_detections), device=device
    )

    i = 0
    for perm in permutations(range(max_detections)):
        # note that we multiply is_on_array:
        # we only evaluate the loss if the star is on
        locs_loss_all_perm[:, i] = (
            locs_log_probs_all[:, perm, :].diagonal(dim1=1, dim2=2) * is_on_array
        ).sum(1)

        # if star, evaluate the star parameters,
        # hence the multiplication by (1 - true_galaxy_bool)
        star_params_loss_all_perm[:, i] = (
            star_params_log_probs_all[:, perm].diagonal(dim1=1, dim2=2)
            * is_on_array
            * (1 - true_galaxy_bool)
        ).sum(1)

        # similarly for galaxies
        galaxy_params_loss_all_perm[:, i] = (
            galaxy_params_log_probs_all[:, perm].diagonal(dim1=1, dim2=2)
            * is_on_array
            * true_galaxy_bool
        ).sum(1)
        i += 1

    return locs_loss_all_perm, star_params_loss_all_perm, galaxy_params_loss_all_perm


def _get_min_perm_loss(
    locs_log_probs_all,
    star_params_log_probs_all,
    galaxy_params_log_probs_all,
    true_galaxy_bool,
    is_on_array,
):

    # get log-probability under every possible matching of estimated star to true star
    (
        locs_log_probs_all_perm,
        star_params_loss_all_perm,
        galaxy_params_loss_all_perm,
    ) = _get_log_probs_all_perms(
        locs_log_probs_all,
        star_params_log_probs_all,
        galaxy_params_log_probs_all,
        is_on_array,
        true_galaxy_bool,
    )

    # find the permutation that minimizes the location losses
    locs_loss, indx = torch.min(-locs_log_probs_all_perm, dim=1)

    # get the star and galaxy losses according to the found permutation
    star_params_loss = -torch.gather(
        star_params_loss_all_perm, 1, indx.unsqueeze(1)
    ).squeeze()
    galaxy_params_loss = -torch.gather(
        galaxy_params_loss_all_perm, 1, indx.unsqueeze(1)
    ).squeeze()

    return locs_loss, star_params_loss, galaxy_params_loss, indx
