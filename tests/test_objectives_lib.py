#!/usr/bin/env python3

import unittest

import torch

import objectives_lib
import star_datasets_lib
import starnet_vae_lib

import json

# Get data that we will use in our tests
with open('./data/default_star_parameters.json', 'r') as fp:
    data_params = json.load(fp)

data_params['min_stars'] = 0
data_params['max_stars'] = 4

# draw data
n_stars = 64
psf_fit_file = \
    './../celeste_net/sdss_stage_dir/3900/6/269/psField-003900-6-0269.fit'

star_dataset = \
    star_datasets_lib.load_dataset_from_params(psf_fit_file,
                            data_params,
                            n_stars = n_stars,
                            use_fresh_data = False,
                            add_noise = True)

class TestStarCounterObjective(unittest.TestCase):
    def test_get_one_hot(self):
        # This tests the "get_one_hot_encoding_from_int"
        # function. We check that it returns a valid one-hot encoding

        n_classes = data_params['max_stars'] + 1
        z = star_dataset.n_stars

        z_one_hot = objectives_lib.get_one_hot_encoding_from_int(z, n_classes)

        assert all(z_one_hot.sum(1) == 1)
        assert all(z_one_hot.float().max(1)[0] == 1)
        assert all(z_one_hot.float().max(1)[1].float() == z)
