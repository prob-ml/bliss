#!/usr/bin/env python3

import unittest
import torch
import hungarian_alg
import time

class TestHungarian(unittest.TestCase):
    def test_hungarian_alg(self):
        # test my implementation of the hungarian algorithm
        # against enumerating all permutations
        dim = 8
        n_trials = 10
        hungarian_time = torch.zeros(n_trials)
        perm_time = torch.zeros(n_trials)

        print('\ntest hungarian: ')
        for n_trial in range(n_trials):
            X = torch.randn(dim, dim)

            # hungarian algorithm
            t0 = time.time()
            perm1 = hungarian_alg.linear_sum_assignment(X)
            hungarian_time[n_trial] = time.time() - t0

            t0 = time.time()
            perm2 = hungarian_alg.find_min_permutation(X)
            perm_time[n_trial] = time.time() - t0

            assert torch.all(torch.Tensor(perm2) == perm1.float())

        print('mean hungarian time: {0:.06f}sec'.format(torch.mean(hungarian_time)))
        print('mean permutation time: {0:.06f}sec'.format(torch.mean(perm_time)))

if __name__ == '__main__':
    unittest.main()
