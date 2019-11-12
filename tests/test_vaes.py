#!/usr/bin/env python3

import unittest

import torch
import numpy as np

import sys
sys.path.insert(0, './')
sys.path.insert(0, '../')
import starnet_vae_lib

import utils
import image_utils

import json

class TestStarEncoder(unittest.TestCase):
    def test_hidden_indx(self):
        # TODO: test the hidden indices
        assert 1 == 1

    def test_forward(self):
        n_image_stamps = 30
        max_detections = 4
        stamp_slen = 9
        n_bands = 2

        # get encoder
        star_encoder = starnet_vae_lib.StarEncoder(full_slen = 101,
                                                   stamp_slen = stamp_slen,
                                                   step = 2,
                                                   edge_padding = 3,
                                                   n_bands = n_bands,
                                                   max_detections = max_detections)

        star_encoder.eval();

        # simulate image stamps and backgrounds
        image_stamps = torch.randn(n_image_stamps, n_bands, stamp_slen, stamp_slen) + 10.
        background_stamps = torch.randn(n_image_stamps, n_bands, stamp_slen, stamp_slen)
        n_star_stamps = torch.Tensor(np.random.choice(max_detections, n_image_stamps)).type(torch.LongTensor)

        # forward
        logit_loc_mean, logit_loc_logvar, \
            log_flux_mean, log_flux_logvar, log_probs = \
                star_encoder(image_stamps, background_stamps, n_star_stamps)

        # test we have the correct pattern of zeros
        assert ((logit_loc_mean != 0).sum(1)[:, 0] == n_star_stamps).all()
        assert ((logit_loc_mean != 0).sum(1)[:, 1] == n_star_stamps).all()

        assert ((logit_loc_logvar != 0).sum(1)[:, 0] == n_star_stamps).all()
        assert ((logit_loc_logvar != 0).sum(1)[:, 1] == n_star_stamps).all()

        for n in range(n_bands):
            assert ((log_flux_mean[:, :, n] != 0).sum(1) == n_star_stamps).all()
            assert ((log_flux_mean[:, :, n] != 0).sum(1) == n_star_stamps).all()

            assert ((log_flux_logvar[:, :, n] != 0).sum(1) == n_star_stamps).all()
            assert ((log_flux_logvar[:, :, n] != 0).sum(1) == n_star_stamps).all()

        # check pattern of zeros
        is_on_array = utils.get_is_on_from_n_stars(n_star_stamps, star_encoder.max_detections)
        assert torch.all((logit_loc_mean * is_on_array.unsqueeze(2).float()) == logit_loc_mean)
        assert torch.all((logit_loc_logvar * is_on_array.unsqueeze(2).float()) == logit_loc_logvar)


        assert torch.all((log_flux_mean * is_on_array.unsqueeze(2).float()) == log_flux_mean)
        assert torch.all((log_flux_logvar * is_on_array.unsqueeze(2).float()) == log_flux_logvar)

        # we check the variational parameters against the hidden parameters
        # one by one
        h_out = star_encoder._forward_to_last_hidden(image_stamps, background_stamps)

        for i in range(n_image_stamps):
            if(n_star_stamps[i] == 0):
                assert torch.all(logit_loc_mean[i] == 0)
                assert torch.all(logit_loc_logvar[i] == 0)
                assert torch.all(log_flux_mean[i] == 0)
                assert torch.all(log_flux_logvar[i] == 0)
            else:
                n_stars_i = int(n_star_stamps[i])

                assert torch.all(logit_loc_mean[i, 0:n_stars_i, :].flatten() == \
                                    h_out[i, star_encoder.locs_mean_indx_mat[n_stars_i][0:(2 * n_stars_i)]])
                assert torch.all(logit_loc_logvar[i, 0:n_stars_i, :].flatten() == \
                                    h_out[i, star_encoder.locs_var_indx_mat[n_stars_i][0:(2 * n_stars_i)]])

                assert torch.all(log_flux_mean[i, 0:n_stars_i, :].flatten() == \
                                    h_out[i, star_encoder.fluxes_mean_indx_mat[n_stars_i][0:(n_bands * n_stars_i)]])
                assert torch.all(log_flux_logvar[i, 0:n_stars_i, :].flatten() == \
                                    h_out[i, star_encoder.fluxes_var_indx_mat[n_stars_i][0:(n_bands * n_stars_i)]])

        # check probabilities
        # free_probs = torch.zeros(n_image_stamps, max_detections + 1)
        # for i in range(star_encoder.max_detections + 1):
        #     h_out = star_encoder._forward_conditional_nstars(h, i)
        #     free_probs[:, i] = h_out[:, -1]

        # assert torch.all(log_probs == star_encoder.log_softmax(free_probs))

        # test that everything works even when n_stars is None
        logit_loc_mean, logit_loc_logvar, \
            log_flux_mean, log_flux_logvar, log_probs = \
                star_encoder(image_stamps, background_stamps, n_stars = None)

        map_n_stars = torch.argmax(log_probs, dim = 1)

        _logit_loc_mean, _logit_loc_logvar, \
            _log_flux_mean, _log_flux_logvar, _log_probs = \
                star_encoder(image_stamps, background_stamps, n_stars = map_n_stars)

        assert torch.all(logit_loc_mean == _logit_loc_mean)
        assert torch.all(logit_loc_logvar == _logit_loc_logvar)
        assert torch.all(log_flux_mean == _log_flux_mean)
        assert torch.all(log_flux_logvar == _log_flux_logvar)
        assert torch.all(log_probs == _log_probs)

    def test_forward_to_hidden2d(self):


        n_image_stamps = 30
        max_detections = 4
        stamp_slen = 9
        n_bands = 2

        # get encoder
        star_encoder = starnet_vae_lib.StarEncoder(full_slen = 101,
                                                   stamp_slen = stamp_slen,
                                                   step = 2,
                                                   edge_padding = 3,
                                                   n_bands = n_bands,
                                                   max_detections = max_detections)

        star_encoder.eval();

        # simulate image stamps and backgrounds
        n_samples = 10
        image_stamps = torch.randn(n_image_stamps, n_bands, stamp_slen, stamp_slen) + 10.
        background_stamps = torch.randn(n_image_stamps, n_bands, stamp_slen, stamp_slen)
        n_star_stamps_sampled = torch.Tensor(np.random.choice(max_detections, (n_samples, n_image_stamps))).type(torch.LongTensor)

        h = star_encoder._forward_to_last_hidden(image_stamps, background_stamps).detach()
        logit_loc_mean, logit_loc_logvar, \
            log_flux_mean, log_flux_logvar = \
                star_encoder._get_params_from_last_hidden_layer(h, n_star_stamps_sampled)

        # CHECK THAT THIS MATCHES MY OLD PARAMETERS
        for i in range(n_samples):
            logit_loc_mean_i, logit_loc_logvar_i, \
                log_flux_mean_i, log_flux_logvar_i, _ = \
                    star_encoder(image_stamps, background_stamps, n_star_stamps_sampled[i])

            assert torch.all(logit_loc_mean_i == logit_loc_mean[i])
            assert torch.all(logit_loc_logvar_i == logit_loc_logvar[i])
            assert torch.all(log_flux_mean_i == log_flux_mean[i])
            assert torch.all(log_flux_logvar_i == log_flux_logvar[i])

    def test_full_params_from_sampled(self):
        n_samples = 10
        stamp_slen = 9
        max_detections = 4
        n_bands = 2

        # get encoder
        star_encoder = starnet_vae_lib.StarEncoder(full_slen = 101,
                                                   stamp_slen = stamp_slen,
                                                   step = 2,
                                                   edge_padding = 3,
                                                   n_bands = n_bands,
                                                   max_detections = max_detections)

        n_image_stamps = star_encoder.tile_coords.shape[0]

        # draw sampled subimage parameters
        n_stars_sampled = torch.Tensor(np.random.choice(max_detections, (n_samples, n_image_stamps))).type(torch.long)
        is_on_array = utils.get_is_on_from_n_stars_2d(n_stars_sampled,
                                                        max_detections).float()

        subimage_locs_sampled = torch.rand((n_samples, n_image_stamps, max_detections, 2)) * is_on_array.unsqueeze(3)
        subimage_fluxes_sampled = torch.rand((n_samples, n_image_stamps, max_detections, n_bands)) * is_on_array.unsqueeze(3)

        locs_full_image, fluxes_full_image, n_stars_full = \
            star_encoder._get_full_params_from_sampled_params(subimage_locs_sampled,
                                                        subimage_fluxes_sampled)

        # test against individually un-patched parameters
        for i in range(n_samples):
            locs_full_image_i, fluxes_full_image_i, n_stars_i  = \
                image_utils.get_full_params_from_patch_params(subimage_locs_sampled[i],
                                                    subimage_fluxes_sampled[i],
                                                    star_encoder.tile_coords,
                                                    star_encoder.full_slen,
                                                    star_encoder.stamp_slen,
                                                    star_encoder.edge_padding)

            assert torch.all(locs_full_image_i == locs_full_image[i, 0:n_stars_i])
            assert torch.all(fluxes_full_image_i == fluxes_full_image[i, 0:n_stars_i])


if __name__ == '__main__':
    unittest.main()
