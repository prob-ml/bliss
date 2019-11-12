#!/usr/bin/env python3

import unittest

import numpy as np

import torch

import sys
sys.path.insert(0, './')
sys.path.insert(0, '../')
import simulated_datasets_lib
import sdss_dataset_lib

import json

import fitsio

psf_dir = './../multiband_pcat/Data/idR-002583-2-0136/psfs/'
psf_r = fitsio.FITS(psf_dir + 'sdss-002583-2-0136-psf-r.fits')[0].read()
psf_g = fitsio.FITS(psf_dir + 'sdss-002583-2-0136-psf-g.fits')[0].read()
psf_og = np.array([psf_r, psf_g])

with open('./data/default_star_parameters.json', 'r') as fp:
    data_params = json.load(fp)

data_params['slen'] = 31
data_params['mean_stars'] = 10

class TestSDSSDataset(unittest.TestCase):

    def test_fresh_data(self):
        # this checks that we are actually drawing fresh data
        # at each epoch (or not)

        n_images = 32
        # get dataset
        star_dataset = \
            simulated_datasets_lib.load_dataset_from_params(psf_og,
                                    data_params,
                                    n_images = n_images,
                                    sky_intensity = torch.Tensor([686., 1000.]),
                                    add_noise = True)

        # get loader
        batchsize = 8

        loader = torch.utils.data.DataLoader(
                         dataset=star_dataset,
                         batch_size=batchsize)

        #############################################
        # First check: all data should be the same
        #############################################
        images_vec = torch.zeros(5)
        locs_vec = torch.zeros(5)
        fluxes_vec = torch.zeros(5)
        n_stars_vec = torch.zeros(5)
        for i in range(5):
            images_mean = 0
            true_locs_mean = 0
            true_fluxes_mean = 0
            true_n_stars_mean = 0

            for _, data in enumerate(loader):
                true_fluxes = data['fluxes']
                true_locs = data['locs']
                true_n_stars = data['n_stars']
                images = data['image']

                images_mean += images.mean()
                true_locs_mean += true_locs.mean()
                true_fluxes_mean += true_fluxes.mean()
                true_n_stars_mean += true_n_stars.float().mean()

            images_vec[i] = images_mean
            locs_vec[i] = true_locs_mean
            fluxes_vec[i] = true_fluxes_mean
            n_stars_vec[i] = true_fluxes_mean

        assert len(images_vec.unique()) == 1
        assert len(locs_vec.unique()) == 1
        assert len(fluxes_vec.unique()) == 1
        assert len(n_stars_vec.unique()) == 1

        #############################################
        # Second check: by adding the "loader.dataset.set_params_and_images()"
        # command, fresh data should be drawn
        #############################################
        images_vec = torch.zeros(5)
        locs_vec = torch.zeros(5)
        fluxes_vec = torch.zeros(5)
        n_stars_vec = torch.zeros(5)
        for i in range(5):
            images_mean = 0
            true_locs_mean = 0
            true_fluxes_mean = 0
            true_n_stars_mean = 0

            for _, data in enumerate(loader):
                true_fluxes = data['fluxes']
                true_locs = data['locs']
                true_n_stars = data['n_stars']
                images = data['image']

                images_mean += images.mean()
                true_locs_mean += true_locs.mean()
                true_fluxes_mean += true_fluxes.mean()
                true_n_stars_mean += true_n_stars.float().mean()

            images_vec[i] = images_mean
            locs_vec[i] = true_locs_mean
            fluxes_vec[i] = true_fluxes_mean
            n_stars_vec[i] = true_fluxes_mean

            # reset
            loader.dataset.set_params_and_images()

        assert len(images_vec.unique()) == 5
        assert len(locs_vec.unique()) == 5
        assert len(fluxes_vec.unique()) == 5
        assert len(n_stars_vec.unique()) == 5

        #############################################
        # Third check: by setting "use_fresh_data = True"
        # command, fresh data should be drawn
        #############################################
        # n_stars = 64
        # # get dataset
        # star_dataset = \
        #     simulated_datasets_lib.load_dataset_from_params(psf_fit_file,
        #                             data_params,
        #                             n_stars = n_stars,
        #                             use_fresh_data = True,
        #                             add_noise = True)
        #
        # # get loader
        # batchsize = 8
        #
        # loader = torch.utils.data.DataLoader(
        #                  dataset=star_dataset,
        #                  batch_size=batchsize)
        #
        # images_vec = torch.zeros(5)
        # locs_vec = torch.zeros(5)
        # fluxes_vec = torch.zeros(5)
        # n_stars_vec = torch.zeros(5)
        # for i in range(5):
        #     images_mean = 0
        #     true_locs_mean = 0
        #     true_fluxes_mean = 0
        #     true_n_stars_mean = 0
        #
        #     for _, data in enumerate(loader):
        #         true_fluxes = data['fluxes']
        #         true_locs = data['locs']
        #         true_n_stars = data['n_stars']
        #         images = data['image']
        #
        #         images_mean += images.mean()
        #         true_locs_mean += true_locs.mean()
        #         true_fluxes_mean += true_fluxes.mean()
        #         true_n_stars_mean += true_n_stars.mean()
        #
        #     images_vec[i] = images_mean
        #     locs_vec[i] = true_locs_mean
        #     fluxes_vec[i] = true_fluxes_mean
        #     n_stars_vec[i] = true_fluxes_mean
        #     loader.dataset.set_params_and_images()
        #
        # assert len(images_vec.unique()) == 5
        # assert len(locs_vec.unique()) == 5
        # assert len(fluxes_vec.unique()) == 5
        # assert len(n_stars_vec.unique()) == 5




if __name__ == '__main__':
    unittest.main()
