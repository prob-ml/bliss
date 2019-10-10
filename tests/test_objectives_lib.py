#!/usr/bin/env python3

import unittest

import torch

import sys
sys.path.insert(0, './../')

import inv_kl_objective_lib as objectives_lib
import simulated_datasets_lib
import starnet_vae_lib
from hungarian_alg import find_min_col_permutation, run_batch_hungarian_alg_parallel

import json

class TestStarCounterObjective(unittest.TestCase):
    def test_get_one_hot(self):
        # This tests the "get_one_hot_encoding_from_int"
        # function. We check that it returns a valid one-hot encoding

        n_classes = 10
        z = torch.randint(0, 10, (100, ))

        z_one_hot = objectives_lib.get_one_hot_encoding_from_int(z, n_classes)

        assert all(z_one_hot.sum(1) == 1)
        assert all(z_one_hot.float().max(1)[0] == 1)
        assert all(z_one_hot.float().max(1)[1].float() == z.float())

class TestStarEncoderObjective(unittest.TestCase):
    def test_perm_mat(self):
        # this tests the _permute_losses_mat function, make sure
        # it returns the correct perumtation of losses

        # get data
        batchsize = 200

        max_detections = 15
        max_stars = 20

        # some losses
        locs_log_probs_all = torch.randn(batchsize, max_stars, max_detections)

        # some permutation
        is_on_array = torch.rand(batchsize, max_stars) > 1
        is_on_array = is_on_array * (is_on_array.sum(dim = 1) < max_detections).unsqueeze(1)
        perm = run_batch_hungarian_alg_parallel(locs_log_probs_all, is_on_array)

        # get losses according to the found permutation
        perm_losses = objectives_lib._permute_losses_mat(locs_log_probs_all, perm)

        # check it worked
        for i in range(batchsize):
            for j in range(max_stars):
                assert perm_losses[i, j] == locs_log_probs_all[i, j, perm[i, j]]

    def test_get_all_comb_losses(self):
        # this checks that our function to return all combination of losses
        # is correct

        batchsize = 10
        max_detections = 4
        max_stars = 6

        # true parameters
        true_locs = torch.rand(batchsize, max_stars, 2)
        true_fluxes = torch.exp(torch.randn(batchsize, max_stars))

        # estimated parameters
        logit_loc_mean = torch.randn(batchsize, max_detections, 2)
        logit_loc_log_var = torch.randn(batchsize, max_detections, 2)

        log_flux_mean  = torch.randn(batchsize, max_detections)
        log_flux_log_var = torch.randn(batchsize, max_detections)

        # get loss for locations
        locs_log_probs_all = \
            objectives_lib.get_locs_logprob_all_combs(true_locs,
                                    logit_loc_mean, logit_loc_log_var)

        # get loss for fluxes
        flux_log_probs_all = \
            objectives_lib.get_fluxes_logprob_all_combs(true_fluxes, \
                                log_flux_mean, log_flux_log_var)

        # for my sanity
        assert list(locs_log_probs_all.shape) == \
            [batchsize, max_stars, max_detections]
        assert list(flux_log_probs_all.shape) == \
            [batchsize, max_stars, max_detections]

        for i in range(batchsize):
            for j in range(max_stars):
                for k in range(max_detections):
                    flux_loss_ij = \
                        objectives_lib.eval_lognormal_logprob(true_fluxes[i, j],
                                                        log_flux_mean[i, k],
                                                        log_flux_log_var[i, k])

                    assert flux_loss_ij == flux_log_probs_all[i, j, k]

                    locs_loss_ij = \
                        objectives_lib.eval_logitnormal_logprob(true_locs[i, j],
                                                logit_loc_mean[i, k],
                                                logit_loc_log_var[i, k]).sum()

                    assert locs_loss_ij == locs_log_probs_all[i, j, k]

    def test_get_weights(self):

        max_stars = 4

        n_stars = torch.randint(0, max_stars + 1, (100, ))

        # get weights
        weights = starnet_vae_lib.get_weights(n_stars)

        # get weights vector
        one_hot = objectives_lib.get_one_hot_encoding_from_int(n_stars, max(n_stars) + 1)
        weights_vec = objectives_lib.get_weights_vec(one_hot, weights)

        # get counts:
        counts = torch.zeros(max_stars + 1)
        for i in range(max_stars + 1):
            counts[i] = torch.sum(n_stars == i)

        for i in range(max_stars + 1):
            assert len(torch.unique(weights_vec[n_stars == i])) == 1

            x = torch.unique(weights_vec[n_stars == i])
            y = counts.max() / counts[i]

            assert torch.abs(x - y) < 1e-6

if __name__ == '__main__':
    unittest.main()
