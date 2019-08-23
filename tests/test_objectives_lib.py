#!/usr/bin/env python3

import unittest

import torch

import sys
sys.path.insert(0, './../')

import objectives_lib
import star_datasets_lib
import starnet_vae_lib
from hungarian_alg import find_min_col_permutation

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
        max_detections = 4
        locs_log_probs_all = torch.randn(batchsize, max_detections, max_detections)

        # some permutation
        perm = objectives_lib.run_batch_hungarian_alg(locs_log_probs_all,
                        n_stars = torch.ones(batchsize) * max_detections)

        # get losses according to the found permutation
        perm_losses = objectives_lib._permute_losses_mat(locs_log_probs_all, perm)

        # check it worked
        for i in range(batchsize):
            for j in range(max_detections):
                assert perm_losses[i, j] == locs_log_probs_all[i, j, perm[i, j]]

    def test_get_all_comb_losses(self):
        # this checks that our function to return all combination of losses
        # is correct

        batchsize = 200
        max_detections = 4

        # true parameters
        true_locs = torch.rand(batchsize, max_detections, 2)
        true_fluxes = torch.exp(torch.randn(batchsize, max_detections))

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
            [batchsize, max_detections, max_detections]
        assert list(flux_log_probs_all.shape) == \
            [batchsize, max_detections, max_detections]

        for i in range(batchsize):
            for j in range(max_detections):
                for k in range(max_detections):
                    flux_loss_ij = \
                        objectives_lib.eval_lognormal_logprob(true_fluxes[i, k],
                                                        log_flux_mean[i, j],
                                                        log_flux_log_var[i, j])

                    assert flux_loss_ij == flux_log_probs_all[i, j, k]

                    locs_loss_ij = \
                        objectives_lib.eval_logitnormal_logprob(true_locs[i, k],
                                                logit_loc_mean[i, j],
                                                logit_loc_log_var[i, j]).sum()

                    assert locs_loss_ij == locs_log_probs_all[i, j, k]

    def test_batch_hungarian(self):

        dim = 4
        batchsize = 10

        X = torch.randn(batchsize, dim, dim)

        perm1 = objectives_lib.run_batch_hungarian_alg(X,
                    n_stars = torch.ones(batchsize) * dim)

        for i in range(batchsize):
            perm2 = find_min_col_permutation(-X[i])

            assert torch.all(perm1[i, :] == torch.LongTensor(perm2))


if __name__ == '__main__':
    unittest.main()
