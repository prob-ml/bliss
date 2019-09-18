#!/usr/bin/env python3

import unittest

import torch
import numpy as np

import sys
sys.path.insert(0, '../')
import inv_KL_objective_lib as objectives_lib
import simulated_datasets_lib
import starnet_vae_lib

import json

class TestStarEncoder(unittest.TestCase):

    def test_forward(self):
        n_images = 30
        max_detections = 4
        slen = 11

        # get encoder
        star_encoder = starnet_vae_lib.StarEncoder(slen = slen,
                                                n_bands = 1,
                                                max_detections = max_detections)
        star_encoder.eval();

        # simulate images and backgrounds
        images = torch.randn(n_images, 1, slen, slen)
        backgrounds = torch.randn(n_images, 1, slen, slen)
        n_stars = torch.Tensor(np.random.choice(max_detections, n_images)).type(torch.LongTensor)

        # forward
        logit_loc_mean, logit_loc_logvar, \
            log_flux_mean, log_flux_logvar, log_probs = \
                star_encoder(images, backgrounds, n_stars)

        # we check the variational parameters against the hidden parameters
        # one by one
        h = star_encoder._forward_to_pooled_hidden(images, backgrounds)

        for i in range(n_images):
            if(n_stars[i] == 0):
                assert torch.all(logit_loc_mean[i] == 0)
                assert torch.all(logit_loc_logvar[i] == 0)
                assert torch.all(log_flux_mean[i] == 0)
                assert torch.all(log_flux_logvar[i] == 0)
            else:
                n_stars_i = int(n_stars[i])
                h_out = star_encoder._forward_conditional_nstars(h, n_stars_i)

                assert torch.all(logit_loc_mean[i, 0:n_stars_i, :] == \
                                    h_out[i, 0:(2 * n_stars_i)].view(n_stars_i, 2))

                assert torch.all(logit_loc_logvar[i, 0:n_stars_i, :] == \
                                    h_out[i, (2 * n_stars_i):(4 * n_stars_i)].view(n_stars_i, 2))

                assert torch.all(log_flux_mean[i, 0:n_stars_i] == \
                                    h_out[i, (4 * n_stars_i):(5 * n_stars_i)])
                assert torch.all(log_flux_logvar[i, 0:n_stars_i] == \
                                    h_out[i, (5 * n_stars_i):(6 * n_stars_i)])

        # check probabilities
        free_probs = torch.zeros(n_images, max_detections + 1)
        for i in range(star_encoder.max_detections + 1):
            h_out = star_encoder._forward_conditional_nstars(h, i)
            free_probs[:, i] = h_out[:, -1]

        assert torch.all(log_probs == star_encoder.log_softmax(free_probs))

        # test that everything works even when n_stars is None
        logit_loc_mean, logit_loc_logvar, \
            log_flux_mean, log_flux_logvar, log_probs = \
                star_encoder(images, backgrounds, n_stars = None)

        map_n_stars = torch.argmax(log_probs, dim = 1)

        _logit_loc_mean, _logit_loc_logvar, \
            _log_flux_mean, _log_flux_logvar, _log_probs = \
                star_encoder(images, backgrounds, n_stars = map_n_stars)

        assert torch.all(logit_loc_mean == _logit_loc_mean)
        assert torch.all(logit_loc_logvar == _logit_loc_logvar)
        assert torch.all(log_flux_mean == _log_flux_mean)
        assert torch.all(log_flux_logvar == _log_flux_logvar)
        assert torch.all(log_probs == _log_probs)


if __name__ == '__main__':
    unittest.main()
