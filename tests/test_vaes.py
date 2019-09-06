#!/usr/bin/env python3

import unittest

import torch
import numpy as np

import sys
sys.path.insert(0, '../')
import objectives_lib
import simulated_datasets_lib
import starnet_vae_lib

import json

class TestStarEncoder(unittest.TestCase):
    def test_params_from_hidden(self):
        # this checks the collection of parameters from the final layer

        n_images = 100
        max_detections = 4

        # get encoder
        star_encoder = starnet_vae_lib.StarEncoder(slen = 11,
                                                n_bands = 1,
                                                max_detections = max_detections)

        # construct a test matrix, with all 1's for the one detection parameters ,
        #   all 2's for the second detection parameters, etc
        h = torch.zeros(n_images, star_encoder.dim_out_all)
        h[:, -1] = max_detections + 10

        indx = 0
        for i in range(1, max_detections + 1):
            n_params_i = 6 * i + 1
            h[:, indx:(indx + n_params_i)] = i

            indx = indx + n_params_i

        # check I constructed this matrix correctly
        for i in range(1, max_detections + 1):
            n_params_i = 6 * i + 1
            assert torch.sum(h == i) == (n_params_i * n_images)

        assert torch.all(h != 0)
        assert torch.all(h[:,-1] == max_detections + 10)

        # test my indexing of parameters
        for i in range(1, max_detections + 1):
            logit_loc_mean, logit_loc_log_var, \
                log_flux_mean, log_flux_log_var, prob_i = \
                    star_encoder._get_params_from_last_hidden_layer(h, i)

            assert torch.all(logit_loc_mean[:, 0:i, :] == i)
            assert torch.all(logit_loc_log_var[:, 0:i, :] == i)

            assert torch.all(logit_loc_log_var[:, i:, :] == 0)
            assert torch.all(logit_loc_log_var[:, i:, :] == 0)

            assert torch.all(log_flux_mean[:, 0:i] == i)
            assert torch.all(log_flux_log_var[:, 0:i] == i)

            assert torch.all(log_flux_mean[:, i:] == 0)
            assert torch.all(log_flux_log_var[:, i:] == 0)

            assert torch.all(prob_i == i)

    def test_forward(self):
        n_images = 30
        max_detections = 4
        slen = 11

        # get encoder
        star_encoder = starnet_vae_lib.StarEncoder(slen = slen,
                                                n_bands = 1,
                                                max_detections = max_detections)

        # simulate images and backgrounds
        images = torch.randn(n_images, 1, slen, slen)
        backgrounds = torch.randn(n_images, 1, slen, slen)
        n_stars = torch.Tensor(np.random.choice(max_detections, n_images))

        # forward
        logit_loc_mean, logit_loc_logvar, \
            log_flux_mean, log_flux_logvar, log_probs = \
                star_encoder(images, backgrounds, n_stars)

        # we check the variational parameters against the hidden parameters
        h = star_encoder.forward_to_last_hidden(images, backgrounds)

        for i in range(n_images):
            if(n_stars[i] == 0):
                assert torch.all(logit_loc_mean[i] == 0)
                assert torch.all(logit_loc_logvar[i] == 0)
                assert torch.all(log_flux_mean[i] == 0)
                assert torch.all(log_flux_logvar[i] == 0)
            else:
                _logit_loc_mean, _logit_loc_log_var, \
                    _log_flux_mean, _log_flux_log_var, prob_i = \
                        star_encoder._get_params_from_last_hidden_layer(h, int(n_stars[i]))

                assert torch.all(logit_loc_mean[i] == _logit_loc_mean[i])
                assert torch.all(logit_loc_logvar[i] == _logit_loc_log_var[i])
                assert torch.all(log_flux_mean[i] == _log_flux_mean[i])
                assert torch.all(log_flux_logvar[i] == _log_flux_log_var[i])

        # check probabilities
        free_probs = torch.zeros(n_images, star_encoder.max_detections + 1)
        for i in range(0, star_encoder.max_detections + 1):
            if(i == 0):
                prob_i = h[:, -1]
            else:
                _, _, _, _, prob_i = \
                    star_encoder._get_params_from_last_hidden_layer(h, i)

                free_probs[:, i] = prob_i

        assert torch.all(log_probs == star_encoder.soft_max(free_probs))

if __name__ == '__main__':
    unittest.main()
