#!/usr/bin/env python3

import unittest

import torch

import sys
sys.path.insert(0, '../')
import objectives_lib
import simulated_datasets_lib
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
    simulated_datasets_lib.load_dataset_from_params(psf_fit_file,
                            data_params,
                            n_stars = n_stars,
                            use_fresh_data = False,
                            add_noise = True)

class TestStarEncoder(unittest.TestCase):
    def test_params_from_hidden(self):
        # this tests the collection of parameters from the final layer

        # get encoder
        star_encoder = starnet_vae_lib.StarEncoder(slen = data_params['slen'],
                            n_bands = 1,
                            max_detections = data_params['max_stars'])

        # construct a test matrix, with all 1's for the one detection parameters ,
        #   all 2's for the second detection parameters, etc
        h = torch.zeros(n_stars, star_encoder.dim_out_all)

        indx = 0
        for i in range(1, data_params['max_stars'] + 1):
            n_params_i = 6 * i
            h[:, indx:(indx + n_params_i)] = i
            indx = indx + n_params_i

            assert torch.sum(h == i) == (i * 6 * n_stars)

        # test my indexing of parameters
        for i in range(1, data_params['max_stars'] + 1):
            logit_loc_mean, logit_loc_log_var, \
                log_flux_mean, log_flux_log_var = \
                    star_encoder._get_params_from_last_hidden_layer(h, i)

            assert torch.all(logit_loc_mean[:, 0:i, :] == i)
            assert torch.all(logit_loc_log_var[:, 0:i, :] == i)

            assert torch.all(logit_loc_log_var[:, i:, :] == 0)
            assert torch.all(logit_loc_log_var[:, i:, :] == 0)

            assert torch.all(log_flux_mean[:, 0:i] == i)
            assert torch.all(log_flux_log_var[:, 0:i] == i)

            assert torch.all(log_flux_mean[:, i:] == 0)
            assert torch.all(log_flux_log_var[:, i:] == 0)

if __name__ == '__main__':
    unittest.main()
