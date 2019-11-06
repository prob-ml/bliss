#!/usr/bin/env python3

import unittest

import torch
import numpy as np

import sys
sys.path.insert(0, '../')
sys.path.insert(0, './')
import image_utils

from utils import get_is_on_from_n_stars
from simulated_datasets_lib import _draw_pareto_maxed

import json

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class TestImageBatching(unittest.TestCase):
    def test_tile_coords(self):

        # define parameters in full image
        full_slen = 100
        subimage_slen = 10
        step = 9
        edge_padding = 0

        # full image:
        full_images = torch.randn(5, 1, full_slen, full_slen)

        # batch image
        images_batched = image_utils.tile_images(full_images, subimage_slen, step)

        # get tile coordinates
        tile_coords = image_utils.get_tile_coords(full_slen, full_slen, subimage_slen, step)

        n_patches = tile_coords.shape[0]

        for i in range(images_batched.shape[0]):

            b = i // n_patches

            x0 = tile_coords[i % n_patches, 0]
            x1 = tile_coords[i % n_patches, 1]

            foo = full_images[b].squeeze()[x0:(x0 + subimage_slen), x1:(x1 + subimage_slen)]

            assert np.all(images_batched[i].squeeze() == foo)

    def test_full_to_patch_to_full(self):
        # we convert full parameters to patch parameters to full parameters
        # and assert that these are consistent

        # define parameters in full image
        full_slen = 100
        subimage_slen = 10
        step = 9
        edge_padding = 0

        # draw full image parameters
        n_images = 5
        max_stars = 1200
        min_stars = 900

        n_stars = np.random.poisson(1000, n_images)
        n_stars = torch.Tensor(n_stars).clamp(max = max_stars,
                        min = min_stars).type(torch.LongTensor)
        is_on_array = get_is_on_from_n_stars(n_stars, max_stars)

        # draw locations
        locs = torch.rand((n_images, max_stars, 2)).to(device) * \
                is_on_array.unsqueeze(2).float()

        # draw fluxes
        fluxes = _draw_pareto_maxed(100, 1e6, alpha = 0.5,
                                shape = (n_images, max_stars)) * is_on_array.float()

        # tile coordinates
        tile_coords = \
            image_utils.get_tile_coords(full_slen, full_slen,
                                        subimage_slen, step)

        # get patches
        subimage_locs, subimage_fluxes, subimage_n_stars, subimage_is_on_array = \
            image_utils.get_params_in_patches(tile_coords, locs, fluxes,
                                                full_slen, subimage_slen)

        # check we have the correct number and pattern of nonzero entries
        assert torch.all((subimage_locs * subimage_is_on_array.unsqueeze(2).float()) == subimage_locs)
        assert torch.all((subimage_fluxes * subimage_is_on_array.float()) == subimage_fluxes)

        assert torch.all((subimage_locs != 0).view(subimage_locs.shape[0], -1).float().sum(1) == \
                            subimage_n_stars.float() * 2)
        assert torch.all((subimage_fluxes != 0).float().sum(1) == subimage_n_stars.float())

        # now convert to full parameters
        locs2, fluxes2, n_stars2 = \
            image_utils.get_full_params_from_patch_params(subimage_locs,
                                                            subimage_fluxes,
                                                            tile_coords,
                                                            full_slen,
                                                            subimage_slen,
                                                            edge_padding)
        for i in range(n_images):
            fluxes_i = fluxes[i, :]
            fluxes2_i = fluxes2[i, :]

            which_on = fluxes_i > 0
            which_on2 = fluxes2_i > 0

            assert which_on.sum() == which_on2.sum()
            assert which_on.sum() == n_stars[i]

            fluxes_i, indx = fluxes_i[which_on].sort()
            fluxes2_i, indx2 = fluxes2_i[which_on2].sort()

            assert torch.all(fluxes_i == fluxes2_i)

            locs_i = locs[i, which_on][indx]
            locs2_i = locs2[i, which_on2][indx2]

            # print((locs_i - locs2_i).abs().max())
            assert (locs_i - locs2_i).abs().max() < 1e-6



if __name__ == '__main__':
    unittest.main()
