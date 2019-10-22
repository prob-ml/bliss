import torch
import numpy as np

from torch.distributions import normal, categorical

import image_utils

from inv_kl_objective_lib import eval_normal_logprob, get_one_hot_encoding_from_int
from simulated_datasets_lib import get_is_on_from_n_stars

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def sample_normal(mean, logvar):
    return mean + torch.exp(0.5 * logvar) * torch.randn(mean.shape).to(device)

def get_kl_prior_term(mean, logvar):
    # ADD this to the LOSS
    return - 0.5 * (1 + logvar - mean.pow(2) - logvar.exp())

def sample_class_weights(class_weights, n_samples = 1):
    """
    draw a sample from Categorical variable with
    probabilities class_weights
    """

    # draw a sample from Categorical variable with
    # probabilities class_weights

    cat_rv = categorical.Categorical(probs = class_weights)
    return cat_rv.sample((n_samples, )).detach().squeeze()

def get_recon_loss(full_images, full_backgrounds,
                    subimage_locs,
                    subimage_fluxes,
                    tile_coords,
                    subimage_slen,
                    edge_padding,
                    simulator):

    locs_full_image, fluxes_full_image, _ = \
        image_utils.get_full_params_from_patch_params(subimage_locs, subimage_fluxes,
                                                tile_coords,
                                                full_images.shape[-1],
                                                subimage_slen,
                                                edge_padding,
                                                full_images.shape[0])

    recon_means = simulator.draw_image_from_params(locs = locs_full_image,
                                                  fluxes = fluxes_full_image,
                                                  n_stars = torch.sum(fluxes_full_image > 0, dim = 1),
                                                  add_noise = False)


    recon_means = recon_means - simulator.sky_intensity + full_backgrounds
    recon_loss = - eval_normal_logprob(full_images, recon_means, torch.log(recon_means)).view(full_images.shape[0], -1).sum(1)

    return recon_means, recon_loss

def get_loss_cond_nstars(star_encoder, full_images, full_backgrounds, h, n_stars, simulator):
    is_on_array = get_is_on_from_n_stars(n_stars, star_encoder.max_detections)

    # get parameters
    logit_loc_mean, logit_loc_logvar, \
        log_flux_mean, log_flux_logvar = \
            star_encoder._get_params_from_last_hidden_layer(h, n_stars)

    # sample locations
    subimage_locs_sampled = torch.sigmoid(sample_normal(logit_loc_mean, logit_loc_logvar)) * \
                                            is_on_array.unsqueeze(2).float()

    # sample fluxes
    subimage_fluxes_sampled = torch.exp(sample_normal(log_flux_mean, log_flux_logvar)) * \
                        is_on_array.float()

    # get reconstruction loss
    _, recon_loss = get_recon_loss(full_images, full_backgrounds,
                                   subimage_locs_sampled, subimage_fluxes_sampled,
                                   star_encoder.tile_coords,
                                   star_encoder.stamp_slen,
                                   star_encoder.edge_padding,
                                   simulator)

    # get kl prior term: this is for each subimage
    locs_kl_prior_term = get_kl_prior_term(logit_loc_mean, logit_loc_logvar).sum(2).sum(1)
    fluxes_kl_prior_term = get_kl_prior_term(log_flux_mean, log_flux_logvar).sum(1)

    # convert to full image batchsize
    locs_kl_prior_term = locs_kl_prior_term.view(full_images.shape[0], -1).sum(1)
    fluxes_kl_prior_term = fluxes_kl_prior_term.view(full_images.shape[0], -1).sum(1)

    return recon_loss + locs_kl_prior_term + fluxes_kl_prior_term

def get_kl_loss(star_encoder,
                full_images,
                full_backgrounds,
                simulator):

    assert simulator.slen == full_images.shape[-1]

    # extract image_patches patches
    image_stamps = star_encoder.get_image_stamps(full_images,
                                        locs = None,
                                        fluxes = None)[0]

    assert full_backgrounds.shape == full_images.shape
    background_stamps = star_encoder.get_image_stamps(full_backgrounds,
                                    locs = None,
                                    fluxes = None)[0]


    # get variational parameters: for ALL n_stars
    # pass through neural network
    h = star_encoder._forward_to_last_hidden(image_stamps, background_stamps)

    # get log probability
    log_probs = star_encoder._get_logprobs_from_last_hidden_layer(h)
    map_n_stars = log_probs.argmax(1)

    ############################
    # get loss at map n_stars
    ############################
    map_loss = get_loss_cond_nstars(star_encoder, full_images, full_backgrounds, h,
                                    map_n_stars, simulator)

    seq_tensor = torch.Tensor([i for i in range(log_probs.shape[0])]).type(torch.LongTensor)
    log_q = log_probs[seq_tensor, map_n_stars].view(full_images.shape[0], -1).sum(1)

    map_ps_loss = map_loss.detach() * log_q + map_loss

    ###############################
    # sample from complement and recompute loss;
    #   second term of rao-blackwellized gradient
    ###############################
    mask = get_one_hot_encoding_from_int(map_n_stars, star_encoder.max_detections + 1)
    conditional_probs = torch.exp(log_probs) * (1 - mask)
    conditional_probs = conditional_probs / conditional_probs.sum(1, keepdim = True)

    n_stars_sampled = sample_class_weights(conditional_probs).detach()

    sampled_loss = get_loss_cond_nstars(star_encoder, full_images, full_backgrounds, h,
                                                n_stars_sampled, simulator)

    log_q_sampled = log_probs[seq_tensor, n_stars_sampled].view(full_images.shape[0], -1).sum(1)

    ps_loss_sampled = sampled_loss.detach() * log_q_sampled + sampled_loss

    ################################
    # get overall pseudo-loss
    ################################
    p = torch.exp(log_q)
    ps_loss = map_ps_loss * p + ps_loss_sampled * (1 - p)

    return map_loss, ps_loss
