#!/usr/bin/env python3

import numpy as np
import unittest

import torch

import sys
sys.path.insert(0, './../')

import utils

class TestUtils(unittest.TestCase):
    def test_get_one_hot(self):
        # This tests the "get_one_hot_encoding_from_int"
        # function. We check that it returns a valid one-hot encoding

        n_classes = 10
        z = torch.randint(0, 10, (100, ))

        z_one_hot = utils.get_one_hot_encoding_from_int(z, n_classes)

        assert all(z_one_hot.sum(1) == 1)
        assert all(z_one_hot.float().max(1)[0] == 1)
        assert all(z_one_hot.float().max(1)[1].float() == z.float())

    def test_is_on_from_n_stars(self):
        max_stars = 10
        n_stars = torch.Tensor(np.random.choice(max_stars, 5)).type(torch.LongTensor)

        is_on = utils.get_is_on_from_n_stars(n_stars, max_stars)

        assert torch.all(is_on.sum(1) == n_stars)
        assert torch.all(is_on == is_on.sort(1, descending = True)[0])

    def test_is_on_from_n_stars2d(self):
        n_samples = 5
        batchsize = 3
        max_stars = 10

        n_stars = torch.Tensor(np.random.choice(max_stars, (n_samples, batchsize))).type(torch.LongTensor)

        is_on = utils.get_is_on_from_n_stars_2d(n_stars, max_stars)

        for i in range(n_samples):
            assert torch.all(utils.get_is_on_from_n_stars(n_stars[i], max_stars) == is_on[i])

if __name__ == '__main__':
    unittest.main()
