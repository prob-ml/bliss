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

np.random.seed(43534)
_ = torch.manual_seed(24534)

class TestImageBatching(unittest.TestCase):
    def test_tile_coords(self):

        # define parameters in full image
        full_slen = 100
        subimage_slen = 10
        step = 9
        edge_padding = 0
        n_bands = 2

        # full image:
        full_images = torch.randn(5, n_bands, full_slen, full_slen)

        # batch image
        images_batched = image_utils.tile_images(full_images, subimage_slen, step)

        # get tile coordinates
        tile_coords = image_utils.get_tile_coords(full_slen, full_slen, subimage_slen, step)

        n_patches = tile_coords.shape[0]

        for i in range(images_batched.shape[0]):

            b = i // n_patches

            x0 = tile_coords[i % n_patches, 0]
            x1 = tile_coords[i % n_patches, 1]

            foo = full_images[b, :, x0:(x0 + subimage_slen), x1:(x1 + subimage_slen)]

            assert np.all(images_batched[i].squeeze() == foo)

    def test_full_to_patch_to_full(self):
        # we convert full parameters to patch parameters to full parameters
        # and assert that these are consistent

        # define parameters in full image
        full_slen = 100
        subimage_slen = 10
        step = 10
        edge_padding = 0
        n_bands = 2

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
        # fudge factor because sometimes there are ties in the fluxes; this messes up my unnittest
        fudge_factor = torch.randn((n_images, max_stars, n_bands)) * 1e-3
        fluxes = (_draw_pareto_maxed(100, 1e6, alpha = 0.5,
                                shape = (n_images, max_stars, n_bands)) + fudge_factor) * \
                is_on_array.unsqueeze(2).float()

        # tile coordinates
        tile_coords = \
            image_utils.get_tile_coords(full_slen, full_slen,
                                        subimage_slen, step)

        # get patches
        patch_locs, patch_fluxes, patch_n_stars, patch_is_on_array = \
            image_utils.get_params_in_patches(tile_coords, locs, fluxes,
                                                full_slen, subimage_slen)

        # check we have the correct number and pattern of nonzero entries
        assert torch.all((patch_locs * patch_is_on_array.unsqueeze(2).float()) == patch_locs)
        assert torch.all((patch_fluxes * patch_is_on_array.unsqueeze(2).float()) == patch_fluxes)

        assert torch.all((patch_locs != 0).view(patch_locs.shape[0], -1).float().sum(1) == \
                            patch_n_stars.float() * 2)

        assert torch.all((patch_fluxes != 0).view(patch_fluxes.shape[0], -1).float().sum(1) == \
                            patch_n_stars.float() * n_bands)


        # now convert to full parameters
        locs2, fluxes2, n_stars2 = \
            image_utils.get_full_params_from_patch_params(patch_locs,
                                                            patch_fluxes,
                                                            tile_coords,
                                                            full_slen,
                                                            subimage_slen,
                                                            edge_padding)
        for i in range(n_images):
            for b in range(n_bands):
                fluxes_i = fluxes[i, :, b]
                fluxes2_i = fluxes2[i, :, b]

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
                assert len(fluxes_i) == len(torch.unique(fluxes_i))
                assert len(fluxes2_i) == len(torch.unique(fluxes2_i))
                assert (locs_i - locs2_i).abs().max() < 1e-6, (locs_i - locs2_i).abs().max()

    def test_full_to_patch(self):
        # simulate one star on the full image; test it lands in the right patch

        tested = False
        while not tested:
            # define parameters in full image
            full_slen = 100
            subimage_slen = 8
            step = 2
            edge_padding = 3
            n_bands = 2

            # draw full image parameters
            n_images = 100
            max_stars = 10

            n_stars = torch.ones(n_images).type(torch.LongTensor)
            is_on_array = get_is_on_from_n_stars(n_stars, max_stars)

            # draw locations
            locs = torch.rand((n_images, max_stars, 2)).to(device) * \
                    is_on_array.unsqueeze(2).float()

            # fluxes
            fluxes = torch.rand((n_images, max_stars, n_bands))

            # tile coordinates
            tile_coords = \
                image_utils.get_tile_coords(full_slen, full_slen,
                                            subimage_slen, step)

            # get patch parameters
            patch_locs, patch_fluxes, patch_n_stars, patch_is_on_array = \
                image_utils.get_params_in_patches(tile_coords, locs, fluxes,
                                                    full_slen, subimage_slen,
                                                     edge_padding)

            n_patches_per_image = tile_coords.shape[0]
            for i in range(n_images):
                # get patches for that image
                _patch_locs = patch_locs[(i * n_patches_per_image):(i + 1)*n_patches_per_image]
                _patch_fluxes = patch_fluxes[(i * n_patches_per_image):(i + 1)*n_patches_per_image]
                _patch_n_stars = patch_n_stars[(i * n_patches_per_image):(i + 1)*n_patches_per_image]

                which_patch = (locs[i][0][0] * (full_slen - 1) > (tile_coords[:, 0] + edge_padding)) & \
                        (locs[i][0][0] * (full_slen - 1) < (tile_coords[:, 0] + subimage_slen - edge_padding - 1)) & \
                        (locs[i][0][1] * (full_slen - 1) > (tile_coords[:, 1] + edge_padding)) & \
                        (locs[i][0][1] * (full_slen - 1) < (tile_coords[:, 1] + subimage_slen - edge_padding - 1))

                if which_patch.sum() == 0:
                    # star might have landed outside the edge padding
                    continue

                tested = True
                assert which_patch.sum() == 1, 'need to choose step so that tiles are disjoint'
                assert (_patch_locs[which_patch] != 0).all()
                assert (_patch_locs[~which_patch] == 0).all()

                assert _patch_n_stars[which_patch] == 1
                assert (_patch_n_stars[~which_patch] == 0).all()

                patch_x0 = (locs[i][0][0] * (full_slen - 1) - (tile_coords[which_patch, 0] + edge_padding - 0.5)) / \
                        (subimage_slen - 2 * edge_padding)

                patch_x1 = (locs[i][0][1] * (full_slen - 1) - (tile_coords[which_patch, 1] + edge_padding - 0.5)) / \
                            (subimage_slen - 2 * edge_padding)

                assert _patch_locs[which_patch].squeeze()[0] == patch_x0
                assert _patch_locs[which_patch].squeeze()[1] == patch_x1
                assert (fluxes[i, 0, :] == _patch_fluxes[which_patch].squeeze()).all()

        assert tested

    def test_patch_to_full(self):
        # draw one star on a subimage patch; check its mapping to the full
        # image works.

        # define parameters in full image
        full_slen = 101
        subimage_slen = 7
        step = 2
        edge_padding = 2
        n_bands = 2

        max_stars = 4

        # tile coordinates
        tile_coords = \
            image_utils.get_tile_coords(full_slen, full_slen,
                                        subimage_slen, step)

        # get subimage parameters
        patch_locs = torch.zeros(tile_coords.shape[0], max_stars, 2)
        patch_fluxes = torch.zeros(tile_coords.shape[0], max_stars, n_bands)
        patch_n_stars = torch.zeros(tile_coords.shape[0])

        # we add a star in one random subimage
        indx = np.random.choice(tile_coords.shape[0])
        patch_locs[indx, 0, :] = torch.rand(2)
        patch_fluxes[indx, 0, :] = torch.rand(n_bands)
        patch_n_stars[indx] = 1

        locs_full_image, fluxes_full_image, n_stars = \
            image_utils.get_full_params_from_patch_params(patch_locs, patch_fluxes,
                                                tile_coords,
                                                full_slen,
                                                subimage_slen,
                                                edge_padding)

        assert (fluxes_full_image.squeeze() == patch_fluxes[indx, 0, :]).all()
        assert n_stars == 1


        test_loc = (patch_locs[indx, 0, :] * \
                        (subimage_slen - 2 * edge_padding) + \
                        tile_coords[indx, :] + edge_padding - 0.5) / (full_slen - 1)

        assert (test_loc == locs_full_image.squeeze()).all()


        # check this works with negative locs
        patch_locs[indx, 0, :] = torch.Tensor([-0.1, 0.5])
        locs_full_image, fluxes_full_image, n_stars = \
            image_utils.get_full_params_from_patch_params(patch_locs, patch_fluxes,
                                                tile_coords,
                                                full_slen,
                                                subimage_slen,
                                                edge_padding)
        test_loc = (patch_locs[indx, 0, :] * \
                        (subimage_slen - 2 * edge_padding) + \
                        tile_coords[indx, :] + edge_padding - 0.5) / (full_slen - 1)
        assert (test_loc == locs_full_image.squeeze()).all()

        # abd with locs > 1
        patch_locs[indx, 0, :] = torch.Tensor([0.1, 1.3])
        locs_full_image, fluxes_full_image, n_stars = \
            image_utils.get_full_params_from_patch_params(patch_locs, patch_fluxes,
                                                tile_coords,
                                                full_slen,
                                                subimage_slen,
                                                edge_padding)
        test_loc = (patch_locs[indx, 0, :] * \
                        (subimage_slen - 2 * edge_padding) + \
                        tile_coords[indx, :] + edge_padding - 0.5) / (full_slen - 1)
        assert (test_loc == locs_full_image.squeeze()).all()



if __name__ == '__main__':
    unittest.main()
