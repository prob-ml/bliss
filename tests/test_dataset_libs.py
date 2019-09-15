#!/usr/bin/env python3

import unittest

import torch

import sys
sys.path.insert(0, './')
sys.path.insert(0, '../')
import simulated_datasets_lib
import sdss_dataset_lib

import json

psf_fit_file = './../celeste_net/sdss_stage_dir/3900/6/269/psField-003900-6-0269.fit'
with open('./data/default_star_parameters.json', 'r') as fp:
    data_params = json.load(fp)

class TestSDSSDataset(unittest.TestCase):
    def test_tile_image(self):
        # this tests the tiling of SDSS images into one batch of images

        image = torch.randn(100, 200)

        batched_image, tile_coords = \
            sdss_dataset_lib._tile_image(image, 10, return_tile_coords=True)

        k = 0
        for i in range(10):
            for j in range(20):
                # check images align
                assert torch.all(batched_image[k] == \
                            image[(10*i):(10*(i + 1)), (10*j):(10*(j + 1))])
                k += 1

        for k in range(batched_image.shape[0]):
            # check my coordinates are correct
            i = tile_coords[k, 0]
            j = tile_coords[k, 1]

            assert torch.all(batched_image[k] == image[i:(i + 10), j:(j + 10)])

    def test_fresh_data(self):
        # this checks that we are actually drawing fresh data
        # at each epoch (or not)

        n_stars = 64
        # get dataset
        star_dataset = \
            simulated_datasets_lib.load_dataset_from_params(psf_fit_file,
                                    data_params,
                                    n_stars = n_stars,
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
