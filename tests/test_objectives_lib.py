#!/usr/bin/env python3

import unittest

import torch
import numpy as np

import sys
sys.path.insert(0, './')
sys.path.insert(0, './../')

import inv_kl_objective_lib as inv_kl_lib
import simulated_datasets_lib
import starnet_vae_lib
import utils

from itertools import permutations

import json

class TestStarEncoderObjective(unittest.TestCase):

    def test_get_all_comb_losses(self):
        # this checks that our function to return all combination of losses
        # is correct

        batchsize = 10
        max_detections = 4
        max_stars = 6
        n_bands = 2

        # true parameters
        true_locs = torch.rand(batchsize, max_stars, 2)
        true_fluxes = torch.exp(torch.randn(batchsize, max_stars, n_bands))

        # estimated parameters
        logit_loc_mean = torch.randn(batchsize, max_detections, 2)
        logit_loc_log_var = torch.randn(batchsize, max_detections, 2)

        log_flux_mean  = torch.randn(batchsize, max_detections, n_bands)
        log_flux_log_var = torch.randn(batchsize, max_detections, n_bands)

        # get loss for locations
        locs_log_probs_all = \
            inv_kl_lib.get_locs_logprob_all_combs(true_locs,
                                    logit_loc_mean, logit_loc_log_var)

        # get loss for fluxes
        flux_log_probs_all = \
            inv_kl_lib.get_fluxes_logprob_all_combs(true_fluxes, \
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
                        utils.eval_lognormal_logprob(true_fluxes[i, j],
                                                        log_flux_mean[i, k],
                                                        log_flux_log_var[i, k]).sum()

                    assert flux_loss_ij == flux_log_probs_all[i, j, k]

                    locs_loss_ij = \
                        utils.eval_logitnormal_logprob(true_locs[i, j],
                                                logit_loc_mean[i, k],
                                                logit_loc_log_var[i, k]).sum()

                    assert locs_loss_ij == locs_log_probs_all[i, j, k]

    def test_get_min_perm_loss(self):

        batchsize = 100
        max_detections = 4
        max_stars = 4
        n_bands = 2

        # true parameters
        n_stars = torch.Tensor(np.random.choice(max_detections + 1, batchsize))
        is_on_array = utils.get_is_on_from_n_stars(n_stars, max_detections).float()

        true_locs = torch.rand(batchsize, max_detections, 2) * is_on_array.unsqueeze(2)
        true_fluxes = torch.exp(torch.randn(batchsize, max_detections, n_bands)) * is_on_array.unsqueeze(2)


        # estimated parameters
        logit_loc_mean = torch.randn(batchsize, max_detections, 2) * is_on_array.unsqueeze(2)
        logit_loc_log_var = torch.randn(batchsize, max_detections, 2) * is_on_array.unsqueeze(2)

        log_flux_mean  = torch.randn(batchsize, max_detections, n_bands) * is_on_array.unsqueeze(2)
        log_flux_log_var = torch.randn(batchsize, max_detections, n_bands) * is_on_array.unsqueeze(2)

        # get loss for locations
        locs_log_probs_all = \
            inv_kl_lib.get_locs_logprob_all_combs(true_locs,
                                    logit_loc_mean, logit_loc_log_var)

        # get loss for fluxes
        flux_log_probs_all = \
            inv_kl_lib.get_fluxes_logprob_all_combs(true_fluxes, \
                                log_flux_mean, log_flux_log_var)



        locs_loss, fluxes_loss, _ = inv_kl_lib.get_min_perm_loss(locs_log_probs_all,
                                    flux_log_probs_all, is_on_array)

        # a quick check for zer0 and one stars
        assert (locs_loss[n_stars == 0] == 0).all()
        assert (locs_loss[n_stars == 1] == -locs_log_probs_all[n_stars == 1][:, 0, 0]).all()
        assert (fluxes_loss[n_stars == 1] == -flux_log_probs_all[n_stars == 1][:, 0, 0]).all()

        # a more thorough check for all possible n_stars
        for i in range(batchsize):
            _n_stars = int(n_stars[i])

            if n_stars[i] == 0:
                assert locs_loss[i] == 0
                continue

            _true_locs = true_locs[i, 0:_n_stars, :]
            _logit_loc_mean = logit_loc_mean[i, 0:_n_stars, :]
            _logit_loc_log_var = logit_loc_log_var[i, 0:_n_stars, :]

            _true_fluxes = true_fluxes[i, 0:_n_stars, :]
            _log_flux_mean = log_flux_mean[i, 0:_n_stars, :]
            _log_flux_log_var = log_flux_log_var[i, 0:_n_stars, :]

            min_locs_loss = 1e16
            for perm in permutations(range(_n_stars)):
                locs_loss_perm = -utils.eval_logitnormal_logprob(_true_locs,
                                    _logit_loc_mean[perm, :], _logit_loc_log_var[perm, :])

                if locs_loss_perm.sum() < min_locs_loss:
                    min_locs_loss = locs_loss_perm.sum()
                    min_fluxes_loss = -utils.eval_lognormal_logprob(_true_fluxes,
                                        _log_flux_mean[perm, :],
                                        _log_flux_log_var[perm, :]).sum()

            assert torch.abs(locs_loss[i] - min_locs_loss) < 1e-5, \
                    torch.abs(locs_loss[i] - min_locs_loss)
            assert torch.abs(fluxes_loss[i] - min_fluxes_loss) < 1e-5, \
                    torch.abs(fluxes_loss[i] - min_fluxes_loss)

        # locs_log_probs_all_perm, fluxes_log_probs_all_perm = \
        #     inv_kl_lib._get_log_probs_all_perms(locs_log_probs_all, flux_log_probs_all, is_on_array)
        # print(locs_log_probs_all_perm[0].argmax())
        #
        # perm_list = []
        # for perm in permutations(range(max_detections)):
        #     perm_list.append(perm)
        #
        # print(perm_list[locs_log_probs_all_perm[0].argmax()])
        #
        #




    #
    # def test_perm_mat(self):
    #     # this tests the _permute_losses_mat function, make sure
    #     # it returns the correct perumtation of losses
    #
    #     # get data
    #     batchsize = 200
    #
    #     max_detections = 15
    #     max_stars = 20
    #
    #     # some losses
    #     locs_log_probs_all = torch.randn(batchsize, max_stars, max_detections)
    #
    #     # some permutation
    #     is_on_array = torch.rand(batchsize, max_stars) > 1
    #     is_on_array = is_on_array * (is_on_array.sum(dim = 1) < max_detections).unsqueeze(1)
    #     perm = run_batch_hungarian_alg_parallel(locs_log_probs_all, is_on_array)
    #
    #     # get losses according to the found permutation
    #     perm_losses = inv_kl_lib._permute_losses_mat(locs_log_probs_all, perm)
    #
    #     # check it worked
    #     for i in range(batchsize):
    #         for j in range(max_stars):
    #             assert perm_losses[i, j] == locs_log_probs_all[i, j, perm[i, j]]


    # def test_get_weights(self):
    #
    #     max_stars = 4
    #
    #     n_stars = torch.randint(0, max_stars + 1, (100, ))
    #
    #     # get weights
    #     weights = starnet_vae_lib.get_weights(n_stars)
    #
    #     # get weights vector
    #     one_hot = inv_kl_lib.get_one_hot_encoding_from_int(n_stars, max(n_stars) + 1)
    #     weights_vec = inv_kl_lib.get_weights_vec(one_hot, weights)
    #
    #     # get counts:
    #     counts = torch.zeros(max_stars + 1)
    #     for i in range(max_stars + 1):
    #         counts[i] = torch.sum(n_stars == i)
    #
    #     for i in range(max_stars + 1):
    #         assert len(torch.unique(weights_vec[n_stars == i])) == 1
    #
    #         x = torch.unique(weights_vec[n_stars == i])
    #         y = counts.max() / counts[i]
    #
    #         assert torch.abs(x - y) < 1e-6
    #
if __name__ == '__main__':
    unittest.main()
