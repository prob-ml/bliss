#!/usr/bin/env python3

import numpy as np

import sys
sys.path.insert(0, '../')

import unittest
import torch
import hungarian_alg
import hungarian_alg_pytorch
import time

class TestHungarian(unittest.TestCase):
    def test_hungarian_alg_pytorch(self):
        # test my implementation of the hungarian algorithm
        # against enumerating all permutations
        dim = 6
        n_trials = 10
        hungarian_time = torch.zeros(n_trials)
        perm_time = torch.zeros(n_trials)

        print('\ntest hungarian: ')
        for n_trial in range(n_trials):
            X = torch.randn(dim, dim)

            # hungarian algorithm
            t0 = time.time()
            perm1 = hungarian_alg_pytorch.linear_sum_assignment(X)
            hungarian_time[n_trial] = time.time() - t0

            t0 = time.time()
            perm2 = hungarian_alg.find_min_col_permutation(X)
            perm_time[n_trial] = time.time() - t0

            assert torch.all(torch.Tensor(perm2) == perm1.float())

        print('mean hungarian time: {0:.06f}sec'.format(torch.mean(hungarian_time)))
        print('mean permutation time: {0:.06f}sec'.format(torch.mean(perm_time)))

    def test_batch_hungarian(self):

        dim = 4
        batchsize = 10

        X = torch.randn(batchsize, dim, dim)

        perm1 = hungarian_alg.run_batch_hungarian_alg(X,
                    n_stars = torch.ones(batchsize) * dim)

        for i in range(batchsize):
            perm2 = hungarian_alg.find_min_col_permutation(-X[i])

            assert torch.all(perm1[i, :] == torch.LongTensor(perm2))

    def test_batch_hungarian_parallel(self):

        dim = 4
        batchsize = 10

        X = torch.randn(batchsize, dim, dim)

        perm1 = hungarian_alg.run_batch_hungarian_alg_parallel(X,
                    n_stars = torch.ones(batchsize) * dim)

        for i in range(batchsize):
            perm2 = hungarian_alg.find_min_col_permutation(-X[i])

            assert torch.all(perm1[i, :] == torch.LongTensor(perm2))



if __name__ == '__main__':
    unittest.main()
