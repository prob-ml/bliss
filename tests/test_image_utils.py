import torch
import numpy as np

from celeste import device
from celeste.models import sourcenet
from celeste.datasets import simulated_datasets


class TestImageBatching:
    def test_tile_coords(self):
        """
        Check that tiled images returned from `sourcenet.tile_images` actually corresponds
        to tiles of the full image.
        """

        # define parameters in full image
        full_slen = 100
        subimage_slen = 10
        step = 9
        edge_padding = 0
        n_bands = 2

        # full image:
        full_images = torch.randn(5, n_bands, full_slen, full_slen)

        # batch image
        images_batched = sourcenet._tile_images(full_images, subimage_slen, step)

        # get tile coordinates
        tile_coords = sourcenet._get_ptile_coords(
            full_slen, full_slen, subimage_slen, step
        )

        n_tiles = tile_coords.shape[0]

        for i in range(images_batched.shape[0]):

            b = i // n_tiles

            x0 = tile_coords[i % n_tiles, 0]
            x1 = tile_coords[i % n_tiles, 1]

            foo = full_images[
                b, :, x0 : (x0 + subimage_slen), x1 : (x1 + subimage_slen)
            ]

            assert torch.all(images_batched[i].squeeze() == foo)

    def test_full_to_tile_to_full(self):
        # we convert full parameters to tile parameters to full parameters
        # and assert that these are consistent

        np.random.seed(43534)
        torch.manual_seed(24534)

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
        n_stars = (
            torch.from_numpy(n_stars)
            .clamp(max=max_stars, min=min_stars)
            .type(torch.LongTensor)
        ).to(device)

        is_on_array = simulated_datasets.get_is_on_from_n_sources(n_stars, max_stars)

        # draw locations
        locs = (
            torch.rand(n_images, max_stars, 2, device=device)
            * is_on_array.unsqueeze(2).float()
        )

        # draw fluxes
        # fudge factor because sometimes there are ties in the fluxes; this messes up unit test.
        fudge_factor = torch.randn(n_images, max_stars, n_bands, device=device) * 1e-3

        fluxes = (
            simulated_datasets._draw_pareto_maxed(
                100, 1e6, alpha=0.5, shape=(n_images, max_stars, n_bands)
            )
            + fudge_factor
        ) * is_on_array.unsqueeze(2).float()

        # tile coordinates
        tile_coords = sourcenet._get_ptile_coords(
            full_slen, full_slen, subimage_slen, step
        )

        # get tiles
        (
            tile_locs,
            tile_fluxes,
            tile_n_stars,
            tile_is_on_array,
        ) = sourcenet._get_params_in_tiles(
            tile_coords, locs, fluxes, full_slen, subimage_slen
        )

        # check we have the correct number and pattern of nonzero entries
        assert torch.all(
            (tile_locs * tile_is_on_array.unsqueeze(2).float()) == tile_locs
        )
        assert torch.all(
            (tile_fluxes * tile_is_on_array.unsqueeze(2).float()) == tile_fluxes
        )

        assert torch.all(
            (tile_locs != 0).view(tile_locs.shape[0], -1).float().sum(1)
            == tile_n_stars.float() * 2
        )

        assert torch.all(
            (tile_fluxes != 0).view(tile_fluxes.shape[0], -1).float().sum(1)
            == tile_n_stars.float() * n_bands
        )

        # now convert to full parameters
        locs2, fluxes2, n_stars2 = sourcenet.get_full_params_from_tile_params(
            tile_locs, tile_fluxes, tile_coords, full_slen, subimage_slen, edge_padding,
        )
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
                assert (locs_i - locs2_i).abs().max() < 1e-6, (
                    (locs_i - locs2_i).abs().max()
                )

    def test_full_to_tile(self):
        # simulate one star on the full image; test it lands in the right tile

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

            n_stars = torch.ones(n_images, dtype=torch.long, device=device)
            is_on_array = simulated_datasets.get_is_on_from_n_sources(
                n_stars, max_stars
            )

            # draw locations
            locs = (
                torch.rand(n_images, max_stars, 2, device=device)
                * is_on_array.unsqueeze(2).float()
            )

            # fluxes
            fluxes = torch.rand(n_images, max_stars, n_bands, device=device)

            # tile coordinates
            tile_coords = sourcenet._get_ptile_coords(
                full_slen, full_slen, subimage_slen, step
            )

            # get tile parameters
            (
                tile_locs,
                tile_fluxes,
                tile_n_stars,
                tile_is_on_array,
            ) = sourcenet._get_params_in_tiles(
                tile_coords, locs, fluxes, full_slen, subimage_slen, edge_padding
            )

            n_tiles_per_image = tile_coords.shape[0]
            for i in range(n_images):
                # get tiles for that image
                _tile_locs = tile_locs[
                    (i * n_tiles_per_image) : (i + 1) * n_tiles_per_image
                ]
                _tile_fluxes = tile_fluxes[
                    (i * n_tiles_per_image) : (i + 1) * n_tiles_per_image
                ]
                _tile_n_stars = tile_n_stars[
                    (i * n_tiles_per_image) : (i + 1) * n_tiles_per_image
                ]

                which_tile = (
                    (
                        locs[i][0][0] * (full_slen - 1)
                        > (tile_coords[:, 0] + edge_padding)
                    )
                    & (
                        locs[i][0][0] * (full_slen - 1)
                        < (tile_coords[:, 0] + subimage_slen - edge_padding - 1)
                    )
                    & (
                        locs[i][0][1] * (full_slen - 1)
                        > (tile_coords[:, 1] + edge_padding)
                    )
                    & (
                        locs[i][0][1] * (full_slen - 1)
                        < (tile_coords[:, 1] + subimage_slen - edge_padding - 1)
                    )
                )

                if which_tile.sum() == 0:
                    # star might have landed outside the edge padding
                    continue

                tested = True
                assert (
                    which_tile.sum() == 1
                ), "need to choose step so that tiles are disjoint"
                assert (_tile_locs[which_tile] != 0).all()
                assert (_tile_locs[~which_tile] == 0).all()

                assert _tile_n_stars[which_tile] == 1
                assert (_tile_n_stars[~which_tile] == 0).all()

                tile_x0 = (
                    locs[i][0][0] * (full_slen - 1)
                    - (tile_coords[which_tile, 0] + edge_padding - 0.5)
                ) / (subimage_slen - 2 * edge_padding)

                tile_x1 = (
                    locs[i][0][1] * (full_slen - 1)
                    - (tile_coords[which_tile, 1] + edge_padding - 0.5)
                ) / (subimage_slen - 2 * edge_padding)

                assert _tile_locs[which_tile].squeeze()[0] == tile_x0
                assert _tile_locs[which_tile].squeeze()[1] == tile_x1
                assert (fluxes[i, 0, :] == _tile_fluxes[which_tile].squeeze()).all()

        assert tested

    def test_tile_to_full(self):
        # draw one star on a subimage tile; check its mapping to the full
        # image works.

        # define parameters in full image
        full_slen = 101
        subimage_slen = 7
        step = 2
        edge_padding = 2
        n_bands = 2

        max_stars = 4

        # tile coordinates
        tile_coords = sourcenet._get_ptile_coords(
            full_slen, full_slen, subimage_slen, step
        )

        # get subimage parameters
        tile_locs = torch.zeros(tile_coords.shape[0], max_stars, 2, device=device)
        tile_fluxes = torch.zeros(
            tile_coords.shape[0], max_stars, n_bands, device=device
        )
        tile_n_stars = torch.zeros(tile_coords.shape[0], device=device)

        # we add a star in one random subimage
        indx = np.random.choice(tile_coords.shape[0])
        tile_locs[indx, 0, :] = torch.rand(2)
        tile_fluxes[indx, 0, :] = torch.rand(n_bands)
        tile_n_stars[indx] = 1

        (
            locs_full_image,
            fluxes_full_image,
            n_stars,
        ) = sourcenet.get_full_params_from_tile_params(
            tile_locs, tile_fluxes, tile_coords, full_slen, subimage_slen, edge_padding,
        )

        assert (fluxes_full_image.squeeze() == tile_fluxes[indx, 0, :]).all()
        assert n_stars == 1

        test_loc = (
            tile_locs[indx, 0, :] * (subimage_slen - 2 * edge_padding)
            + tile_coords[indx, :]
            + edge_padding
            - 0.5
        ) / (full_slen - 1)

        assert torch.all(test_loc.eq(locs_full_image.squeeze()))

        # check this works with negative locs
        tile_locs[indx, 0, :] = torch.from_numpy(np.array([-0.1, 0.5]))
        (
            locs_full_image,
            fluxes_full_image,
            n_stars,
        ) = sourcenet.get_full_params_from_tile_params(
            tile_locs, tile_fluxes, tile_coords, full_slen, subimage_slen, edge_padding,
        )
        test_loc = (
            tile_locs[indx, 0, :] * (subimage_slen - 2 * edge_padding)
            + tile_coords[indx, :]
            + edge_padding
            - 0.5
        ) / (full_slen - 1)
        assert (test_loc == locs_full_image.squeeze()).all()

        # abd with locs > 1
        tile_locs[indx, 0, :] = torch.from_numpy(np.array([0.1, 1.3]))
        (
            locs_full_image,
            fluxes_full_image,
            n_stars,
        ) = sourcenet.get_full_params_from_tile_params(
            tile_locs, tile_fluxes, tile_coords, full_slen, subimage_slen, edge_padding,
        )
        test_loc = (
            tile_locs[indx, 0, :] * (subimage_slen - 2 * edge_padding)
            + tile_coords[indx, :]
            + edge_padding
            - 0.5
        ) / (full_slen - 1)

        assert (test_loc == locs_full_image.squeeze()).all()
