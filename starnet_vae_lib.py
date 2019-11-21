import torch
import torch.nn as nn

import numpy as np

import image_utils
import utils

from torch.distributions import poisson

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Flatten(nn.Module):
    def forward(self, tensor):
        return tensor.view(tensor.size(0), -1)

class Normalize2d(nn.Module):
    def forward(self, tensor):
        assert len(tensor.shape) == 4
        mean = tensor.view(tensor.shape[0], tensor.shape[1], -1).mean(2, keepdim = True).unsqueeze(-1)
        var = tensor.view(tensor.shape[0], tensor.shape[1], -1).var(2, keepdim = True).unsqueeze(-1)

        return (tensor - mean) / torch.sqrt(var + 1e-5)


class StarEncoder(nn.Module):
    def __init__(self, full_slen, stamp_slen, step, edge_padding, n_bands, max_detections):

        super(StarEncoder, self).__init__()

        # image parameters
        self.full_slen = full_slen # dimension of full image: we assume its square for now
        self.stamp_slen = stamp_slen # dimension of the individual image patches
        self.step = step # number of pixels to shift every subimage
        self.n_bands = n_bands

        self.edge_padding = edge_padding

        self.tile_coords = image_utils.get_tile_coords(full_slen, full_slen,
                                                        stamp_slen, step);
        self.n_patches = self.tile_coords.shape[0]


        # TODO: make this variable for mean stars
        mean_stars_per_patch = 0.4
        poisson_dstr = poisson.Poisson(rate = mean_stars_per_patch)
        inv_weights = torch.exp(poisson_dstr.log_prob(torch.arange(max_detections + 1).float()))

        self.weights = (inv_weights.max() / inv_weights).to(device)

        # max number of detections
        self.max_detections = max_detections

        # convolutional NN paramters
        enc_conv_c = 20
        enc_kern = 3
        enc_hidden = 256

        momentum = 0.5

        # convolutional NN
        self.enc_conv = nn.Sequential(
            nn.Conv2d(self.n_bands, enc_conv_c, enc_kern,
                        stride=1, padding=1),
            nn.ReLU(),

            nn.Conv2d(enc_conv_c, enc_conv_c, enc_kern,
                        stride=1, padding=1),
            nn.BatchNorm2d(enc_conv_c, momentum=momentum, track_running_stats=True),
            nn.ReLU(),

            nn.Conv2d(enc_conv_c, enc_conv_c, enc_kern,
                        stride=1, padding=1),
            nn.ReLU(),

            nn.Conv2d(enc_conv_c, enc_conv_c, enc_kern,
                        stride=1, padding=1),
            nn.BatchNorm2d(enc_conv_c, momentum=momentum, track_running_stats=True),
            nn.ReLU(),
            Flatten()
        )

        # output dimension of convolutions
        conv_out_dim = \
            self.enc_conv(torch.zeros(1, n_bands, stamp_slen, stamp_slen)).size(1)

        # fully connected layers
        self.enc_fc = nn.Sequential(
            nn.Linear(conv_out_dim, enc_hidden),
            nn.BatchNorm1d(enc_hidden, momentum=momentum, track_running_stats=True),
            nn.ReLU(),

            nn.Linear(enc_hidden, enc_hidden),
            nn.BatchNorm1d(enc_hidden, momentum=momentum, track_running_stats=True),
            nn.ReLU(),

            nn.Linear(enc_hidden, enc_hidden),
            nn.BatchNorm1d(enc_hidden, momentum=momentum, track_running_stats=True),
            nn.ReLU(),
        )

        # add final layer, whose size depends on the number of stars to output
        # for i in range(0, max_detections + 1):
        #     # i = 0, 1, ..., max_detections
        #     len_out = i * 6 + 1
        #     width_hidden = len_out * 10
        #
        #     module_a = nn.Sequential(nn.Linear(enc_hidden, width_hidden),
        #                             nn.ReLU(),
        #                             nn.Linear(width_hidden, width_hidden),
        #                             # nn.ReLU(),
        #                             # nn.Linear(width_hidden, width_hidden),
        #                             nn.ReLU())
        #     self.add_module('enc_a_detect' + str(i), module_a)
        #
        #     module_b = nn.Sequential(nn.Linear(width_hidden + enc_hidden, width_hidden),
        #                             nn.ReLU(),
        #                             nn.Linear(width_hidden, width_hidden),
        #                             # nn.ReLU(),
        #                             # nn.Linear(width_hidden, width_hidden),
        #                             nn.ReLU())
        #
        #     self.add_module('enc_b_detect' + str(i), module_b)
        #
        #     final_module_name = 'enc_final_detect' + str(i)
        #     final_module = nn.Linear(2 * width_hidden + enc_hidden, len_out)
        #     self.add_module(final_module_name, final_module)

        # there are self.max_detections * (self.max_detections + 1)
        #    total possible detections, and each detection has
        #    seven parameters (thhee means, three variances, for two locs and one
        #    flux; and one probability)
        self.n_params_per_star = (4 + 2 * self.n_bands)
        self.dim_out_all = \
            int(0.5 * self.max_detections * (self.max_detections + 1) * self.n_params_per_star + \
                    1 + self.max_detections)
        self._get_hidden_indices()

        self.enc_final = nn.Linear(enc_hidden, self.dim_out_all)
        self.log_softmax = nn.LogSoftmax(dim = 1)

    ############################
    # The layers of our neural network
    ############################
    def _forward_to_pooled_hidden(self, image, background):
        # forward to the layer that is shared by all n_stars
        assert torch.all(image > 0.)
        assert torch.all((image - background) > -1000.)
        assert image.shape == background.shape

        log_img = torch.log(image - background + 1000)

        # means = log_img.view(image.shape[0], self.n_bands, -1).mean(-1)
        # stds = log_img.view(image.shape[0], self.n_bands, -1).std(-1)
        # mins = log_img.view(log_img.shape[0], self.n_bands, -1).min(-1)[0]
        # maxes = log_img.view(log_img.shape[0], self.n_bands, -1).max(-1)[0]
        #
        # log_img = (log_img - mins.unsqueeze(-1).unsqueeze(-1)) / (maxes - mins).unsqueeze(-1).unsqueeze(-1)

        h = self.enc_conv(log_img)

        return self.enc_fc(h)

    def _forward_to_last_hidden(self, image_stamps, background_stamps):
        # concatenate all output parameters for all possible n_stars

        h = self._forward_to_pooled_hidden(image_stamps, background_stamps)

        return self.enc_final(h)

        # h_out = torch.zeros(image_stamps.shape[0], 1).to(device)
        # for i in range(0, self.max_detections + 1):
        #     h_i = self._forward_conditional_nstars(h, i)
        #
        #     h_out = torch.cat((h_out, h_i), dim = 1)
        #
        # return h_out[:, 1:h_out.shape[1]]

    ######################
    # Forward modules
    ######################
    def forward(self, image_stamps, background_stamps, n_stars = None):
        # pass through neural network
        h = self._forward_to_last_hidden(image_stamps, background_stamps)

        # get probability of n_stars
        log_probs = self._get_logprobs_from_last_hidden_layer(h)

        if n_stars is None:
            n_stars = torch.argmax(log_probs, dim = 1)

        # extract parameters
        logit_loc_mean, logit_loc_logvar, \
            log_flux_mean, log_flux_logvar = \
                self._get_params_from_last_hidden_layer(h, n_stars)

        return logit_loc_mean, logit_loc_logvar, \
                log_flux_mean, log_flux_logvar, log_probs

    def _get_logprobs_from_last_hidden_layer(self, h):
        free_probs = h[:, self.prob_indx]

        return self.log_softmax(free_probs)

    def _get_params_from_last_hidden_layer(self, h, n_stars):

        if len(n_stars.shape) == 1:
            n_stars = n_stars.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False

        # this class takes in an array of n_stars, n_samples x batchsize
        assert h.shape[1] == self.dim_out_all
        assert h.shape[0] == n_stars.shape[1]

        n_samples = n_stars.shape[0]

        batchsize = h.size(0)
        _h = torch.cat((h, torch.zeros(batchsize, 1).to(device)), dim = 1)

        logit_loc_mean = torch.gather(_h, 1, self.locs_mean_indx_mat[n_stars.transpose(0, 1)].reshape(batchsize, -1))
        logit_loc_logvar = torch.gather(_h, 1, self.locs_var_indx_mat[n_stars.transpose(0, 1)].reshape(batchsize, -1))

        log_flux_mean = torch.gather(_h, 1, self.fluxes_mean_indx_mat[n_stars.transpose(0, 1)].reshape(batchsize, -1))
        log_flux_logvar = torch.gather(_h, 1, self.fluxes_var_indx_mat[n_stars.transpose(0, 1)].reshape(batchsize, -1))

        # reshape
        logit_loc_mean =  logit_loc_mean.reshape(batchsize, n_samples, self.max_detections, 2).transpose(0, 1)
        logit_loc_logvar = logit_loc_logvar.reshape(batchsize, n_samples, self.max_detections, 2).transpose(0, 1)
        log_flux_mean = log_flux_mean.reshape(batchsize, n_samples, self.max_detections, self.n_bands).transpose(0, 1)
        log_flux_logvar = log_flux_logvar.reshape(batchsize, n_samples, self.max_detections, self.n_bands).transpose(0, 1)

        if squeeze_output:
            return logit_loc_mean.squeeze(0), logit_loc_logvar.squeeze(0), \
                        log_flux_mean.squeeze(0), log_flux_logvar.squeeze(0)
        else:
            return logit_loc_mean, logit_loc_logvar, \
                        log_flux_mean, log_flux_logvar

    def _get_hidden_indices(self):

        self.locs_mean_indx_mat = \
            torch.full((self.max_detections + 1, 2 * self.max_detections),
                        self.dim_out_all).type(torch.LongTensor).to(device)

        self.locs_var_indx_mat = \
            torch.full((self.max_detections + 1, 2 * self.max_detections),
                        self.dim_out_all).type(torch.LongTensor).to(device)

        self.fluxes_mean_indx_mat = \
            torch.full((self.max_detections + 1, self.n_bands * self.max_detections),
                        self.dim_out_all).type(torch.LongTensor).to(device)
        self.fluxes_var_indx_mat = \
            torch.full((self.max_detections + 1, self.n_bands * self.max_detections),
                        self.dim_out_all).type(torch.LongTensor).to(device)

        self.prob_indx = torch.zeros(self.max_detections + 1).type(torch.LongTensor).to(device)

        for n_detections in range(1, self.max_detections + 1):
            indx0 = int(0.5 * n_detections * (n_detections - 1) * self.n_params_per_star) + \
                            (n_detections  - 1) + 1

            indx1 = (2 * n_detections) + indx0
            indx2 = (2 * n_detections) * 2 + indx0

            # indices for locations
            self.locs_mean_indx_mat[n_detections, 0:(2 * n_detections)] = torch.arange(indx0, indx1)
            self.locs_var_indx_mat[n_detections, 0:(2 * n_detections)] = torch.arange(indx1, indx2)

            indx3 = indx2 + (n_detections * self.n_bands)
            indx4 = indx3 + (n_detections * self.n_bands)

            # indices for fluxes
            self.fluxes_mean_indx_mat[n_detections, 0:(n_detections * self.n_bands)] = torch.arange(indx2, indx3)
            self.fluxes_var_indx_mat[n_detections, 0:(n_detections * self.n_bands)] = torch.arange(indx3, indx4)

            self.prob_indx[n_detections] = indx4

    ######################
    # Modules for patching images and parameters
    ######################
    def get_image_stamps(self, images_full, locs, fluxes,
                            trim_images = False, clip_max_stars = False):
        assert len(images_full.shape) == 4 # should be batchsize x n_bands x full_slen x full_slen
        assert images_full.shape[1] == self.n_bands
        assert images_full.shape[2] == self.full_slen
        assert images_full.shape[3] == self.full_slen


        batchsize = images_full.shape[0]

        image_stamps = \
            image_utils.tile_images(images_full,
                                    self.stamp_slen,
                                    self.step)

        if (locs is not None) and (fluxes is not None):
            assert fluxes.shape[2] == self.n_bands

            # get parameters in patch as well
            subimage_locs, subimage_fluxes, n_stars, is_on_array = \
                image_utils.get_params_in_patches(self.tile_coords,
                                                  locs,
                                                  fluxes,
                                                  self.full_slen,
                                                  self.stamp_slen,
                                                  self.edge_padding)

            # if (self.weights is None) or (images_full.shape[0] != self.batchsize):
            #     self.weights = get_weights(n_stars.clamp(max = self.max_detections))

            if clip_max_stars:
                n_stars = n_stars.clamp(max = self.max_detections)
                subimage_locs = subimage_locs[:, 0:self.max_detections, :]
                subimage_fluxes = subimage_fluxes[:, 0:self.max_detections, :]
                is_on_array = is_on_array[:, 0:self.max_detections]

        else:
            subimage_locs = None
            subimage_fluxes = None
            n_stars = None
            is_on_array = None

        if trim_images:
            image_stamps = image_utils.trim_images(image_stamps, self.edge_padding)

        return image_stamps, subimage_locs, subimage_fluxes, \
                    n_stars, is_on_array

    ######################
    # Modules to sample our variational distribution and get parameters on the full image
    ######################
    def _get_full_params_from_sampled_params(self, subimage_locs_sampled,
                                                subimage_fluxes_sampled):
        n_samples = subimage_locs_sampled.shape[0]
        n_image_stamps = subimage_locs_sampled.shape[1]

        assert self.n_bands == subimage_fluxes_sampled.shape[-1]

        assert (n_image_stamps % self.tile_coords.shape[0]) == 0

        locs_full_image, fluxes_full_image, n_stars_full = \
            image_utils.get_full_params_from_patch_params(subimage_locs_sampled.view(n_samples * n_image_stamps, -1, 2),
                                                          subimage_fluxes_sampled.view(n_samples * n_image_stamps, -1, self.n_bands),
                                                        self.tile_coords,
                                                        self.full_slen,
                                                        self.stamp_slen,
                                                        self.edge_padding)

        return locs_full_image, fluxes_full_image, n_stars_full

    def sample_star_encoder(self, full_image,
                                full_background,
                                n_samples = 1,
                                return_map = False,
                                n_stars = None,
                                return_log_q = False,
                                training = False):

        # our sampling only works for one image at a time at the moment ...
        assert full_image.shape[0] == 1

        # the image stamps
        image_stamps = self.get_image_stamps(full_image,
                            locs = None, fluxes = None, trim_images = False)[0]
        background_stamps = self.get_image_stamps(full_background,
                            locs = None, fluxes = None, trim_images = False)[0]

        # pass through NN
        h = self._forward_to_last_hidden(image_stamps, background_stamps)

        if not training:
            h = h.detach()

        # get log probs
        log_probs = self._get_logprobs_from_last_hidden_layer(h).detach()

        # sample number of stars
        if n_stars is None:
            if return_map:
                n_stars_sampled = torch.argmax(log_probs, dim = 1).repeat(n_samples).view(n_samples, -1)

            else:
                n_stars_sampled = utils.sample_class_weights(torch.exp(log_probs), n_samples).view(n_samples, -1)
        else:
            n_stars_sampled = n_stars.repeat(n_samples).view(n_samples, -1)

        is_on_array = utils.get_is_on_from_n_stars_2d(n_stars_sampled,
                                self.max_detections)

        # get variational parameters
        logit_loc_mean, logit_loc_logvar, \
            log_flux_mean, log_flux_logvar = \
                self._get_params_from_last_hidden_layer(h, n_stars_sampled)

        if return_map:
            logit_loc_sd = torch.zeros(logit_loc_logvar.shape).to(device)
            log_flux_sd = torch.zeros(log_flux_logvar.shape).to(device)
        else:
            logit_loc_sd = torch.exp(0.5 * logit_loc_logvar)
            log_flux_sd = torch.exp(0.5 * log_flux_logvar)

        # sample locations
        locs_randn = torch.randn(logit_loc_mean.shape).to(device)

        logit_locs_sampled = logit_loc_mean + locs_randn * logit_loc_sd
        subimage_locs_sampled = \
            torch.sigmoid(logit_locs_sampled) * is_on_array.unsqueeze(3).float()

        # sample fluxes
        fluxes_randn = torch.randn(log_flux_mean.shape).to(device);
        log_flux_sampled = log_flux_mean + fluxes_randn * log_flux_sd

        subimage_fluxes_sampled = \
            torch.exp(log_flux_sampled) * is_on_array.unsqueeze(3).float()

        # get parameters on full image
        locs_full_image, fluxes_full_image, n_stars_full = \
            self._get_full_params_from_sampled_params(subimage_locs_sampled,
                                                        subimage_fluxes_sampled)

        if return_log_q:
            log_q_locs = (utils.eval_normal_logprob(logit_locs_sampled, logit_loc_mean,
                                                        logit_loc_logvar) * \
                                                        is_on_array.float().unsqueeze(3)).view(n_samples, -1).sum(1)
            log_q_fluxes = (utils.eval_normal_logprob(log_flux_sampled, log_flux_mean,
                                                log_flux_logvar) * \
                                                is_on_array.float().unsqueeze(3)).view(n_samples, -1).sum(1)
            log_q_n_stars = torch.gather(log_probs, 1, n_stars_sampled.transpose(0, 1)).transpose(0, 1).sum(1)

        else:
            log_q_locs = None
            log_q_fluxes = None
            log_q_n_stars = None

        return locs_full_image, fluxes_full_image, n_stars_full, \
                log_q_locs, log_q_fluxes, log_q_n_stars


# def get_weights(n_stars):
#     weights = torch.zeros(max(n_stars) + 1).to(device)
#     for i in range(max(n_stars) + 1):
#         weights[i] = len(n_stars) / torch.sum(n_stars == i).float()
#     return weights / weights.min()

# def get_results_on_full_image(self, images_full, backgrounds_full,
#                                     true_n_stars = None, n_samples = 0):
#     # first convert full iimages to image stamps
#
#     # get image stamps
#     image_stamps = self.get_image_stamps(images_full, None, None,
#                                           trim_images = False)[0]
#     background_stamps = self.get_image_stamps(backgrounds_full, None, None,
#                                           trim_images = False)[0]
#
#     # get variational parameters on stamps
#     logit_loc_mean, logit_loc_log_var, \
#         log_flux_mean, log_flux_log_var, log_probs = \
#             self.forward(image_stamps, background_stamps, true_n_stars)
#
#     # get map estimates on image stamps
#     if true_n_stars is None:
#         map_n_stars = torch.argmax(log_probs, dim = 1)
#     else:
#         map_n_stars = true_n_stars
#
#     is_on_array = get_is_on_from_n_stars(map_n_stars, self.max_detections)
#
#     map_locs = torch.sigmoid(logit_loc_mean).detach() * is_on_array.unsqueeze(2).float()
#     map_fluxes = torch.exp(log_flux_mean).detach() * is_on_array.float()
#
#     # convert stamp parameters to parameters on the full image
#     map_locs_full_image, map_fluxes_full_image, n_stars_full = \
#         image_utils.get_full_params_from_patch_params(map_locs,
#                                                       map_fluxes,
#                                                     self.tile_coords,
#                                                     self.full_slen,
#                                                     self.stamp_slen,
#                                                     self.edge_padding,
#                                                     self.batchsize)
#
#     if n_samples > 0:
#         # TODO: this is not thoroughly tested ...
#         logit_loc_sample = logit_loc_mean.unsqueeze(3) + \
#                                 torch.randn((logit_loc_mean.shape[0],
#                                             logit_loc_mean.shape[1],
#                                             logit_loc_mean.shape[2],
#                                             n_samples)) * \
#                                 torch.exp(0.5 * logit_loc_log_var).unsqueeze(3)
#
#         locs_sampled = torch.sigmoid(logit_loc_sample) * is_on_array.unsqueeze(2).unsqueeze(3).float()
#
#         log_flux_sample = log_flux_mean.unsqueeze(2) + \
#                                 torch.randn((log_flux_mean.shape[0],
#                                             log_flux_mean.shape[1],
#                                             n_samples)) * \
#                                 torch.exp(0.5 * log_flux_log_var).unsqueeze(2)
#         fluxes_sampled = torch.exp(log_flux_sample) * is_on_array.unsqueeze(2).float()
#
#         locs_sampled_full_image, fluxes_sampled_full_image, _ = \
#             image_utils.get_full_params_from_patch_params(locs_sampled.view(locs_sampled.shape[0], -1, 2),
#                                                           fluxes_sampled.view(fluxes_sampled.shape[0], -1),
#                                                         self.tile_coords,
#                                                         self.full_slen,
#                                                         self.stamp_slen,
#                                                         self.edge_padding,
#                                                         self.batchsize)
#
#         return map_locs_full_image, map_fluxes_full_image, n_stars_full, \
#                 locs_sampled_full_image, fluxes_sampled_full_image
#
#     else:
#         return map_locs_full_image, map_fluxes_full_image, n_stars_full
#

# def _get_params_from_last_hidden_layer(self, h, n_stars):
#
#     assert h.shape[1] == self.dim_out_all
#     assert h.shape[0] == len(n_stars)
#
#     batchsize = h.size(0)
#     _h = torch.cat((h, torch.zeros(batchsize, 1).to(device)), dim = 1)
#
#     logit_loc_mean = torch.gather(_h, 1, self.locs_mean_indx_mat[n_stars])
#     logit_loc_logvar = torch.gather(_h, 1, self.locs_var_indx_mat[n_stars])
#
#     log_flux_mean = torch.gather(_h, 1, self.fluxes_mean_indx_mat[n_stars])
#     log_flux_logvar = torch.gather(_h, 1, self.fluxes_var_indx_mat[n_stars])
#
#     return logit_loc_mean.reshape(batchsize, self.max_detections, 2), \
#             logit_loc_logvar.reshape(batchsize, self.max_detections, 2), \
#             log_flux_mean.reshape(batchsize, self.max_detections), \
#             log_flux_logvar.reshape(batchsize, self.max_detections)
