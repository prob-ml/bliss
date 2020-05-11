import torch
import numpy as np

import celeste.utils
from celeste.models import sourcenet_lib
from celeste import utils


class TestSourceEncoder:
    def test_hidden_indx(self):
        # TODO: test the hidden indices
        assert 1 == 1

    def test_forward(self):
        """
        * Test that forward returns the correct pattern of zeros.
        * Test that variational parameters inside h agree with those returned from forward.
        * Test everything works with n_stars=None in forward.
        """
        n_image_tiles = 30
        max_detections = 4
        ptile_slen = 9
        n_bands = 2

        # get encoder
        star_encoder = sourcenet_lib.SourceEncoder(
            slen=101,
            ptile_slen=ptile_slen,
            step=2,
            edge_padding=3,
            n_bands=n_bands,
            max_detections=max_detections,
            n_source_params=n_bands,
        ).to(utils.device)

        star_encoder.eval()

        # simulate image padded tiles
        image_ptiles = (
            torch.randn(n_image_tiles, n_bands, ptile_slen, ptile_slen) + 10.0
        ).to(utils.device)

        n_star_per_tile = (
            torch.Tensor(np.random.choice(max_detections, n_image_tiles))
            .type(torch.LongTensor)
            .to(utils.device)
        )

        # forward
        (
            loc_mean,
            loc_logvar,
            log_flux_mean,
            log_flux_logvar,
            log_probs,
        ) = star_encoder.forward(image_ptiles, n_star_per_tile)

        assert torch.all(loc_mean <= 1.0)
        assert torch.all(loc_mean >= 0.0)

        # test we have the correct pattern of zeros
        assert ((loc_mean != 0).sum(1)[:, 0] == n_star_per_tile).all()
        assert ((loc_mean != 0).sum(1)[:, 1] == n_star_per_tile).all()

        assert ((loc_logvar != 0).sum(1)[:, 0] == n_star_per_tile).all()
        assert ((loc_logvar != 0).sum(1)[:, 1] == n_star_per_tile).all()

        for n in range(n_bands):
            assert ((log_flux_mean[:, :, n] != 0).sum(1) == n_star_per_tile).all()
            assert ((log_flux_mean[:, :, n] != 0).sum(1) == n_star_per_tile).all()

            assert ((log_flux_logvar[:, :, n] != 0).sum(1) == n_star_per_tile).all()
            assert ((log_flux_logvar[:, :, n] != 0).sum(1) == n_star_per_tile).all()

        # check pattern of zeros
        is_on_array = utils.get_is_on_from_n_sources(
            n_star_per_tile, star_encoder.max_detections
        )
        assert torch.all((loc_mean * is_on_array.unsqueeze(2).float()) == loc_mean)
        assert torch.all((loc_logvar * is_on_array.unsqueeze(2).float()) == loc_logvar)

        assert torch.all(
            (log_flux_mean * is_on_array.unsqueeze(2).float()) == log_flux_mean
        )
        assert torch.all(
            (log_flux_logvar * is_on_array.unsqueeze(2).float()) == log_flux_logvar
        )

        # we check the variational parameters against the hidden parameters
        # one by one
        h_out = star_encoder._get_var_params_all(image_ptiles)

        for i in range(n_image_tiles):
            if n_star_per_tile[i] == 0:
                assert torch.all(loc_mean[i] == 0)
                assert torch.all(loc_logvar[i] == 0)
                assert torch.all(log_flux_mean[i] == 0)
                assert torch.all(log_flux_logvar[i] == 0)
            else:
                n_stars_i = int(n_star_per_tile[i])

                assert torch.all(
                    loc_mean[i, 0:n_stars_i, :].flatten()
                    == torch.sigmoid(h_out)[
                        i,
                        star_encoder.locs_mean_indx_mat[n_stars_i][0 : (2 * n_stars_i)],
                    ]
                )
                assert torch.all(
                    loc_logvar[i, 0:n_stars_i, :].flatten()
                    == h_out[
                        i,
                        star_encoder.locs_var_indx_mat[n_stars_i][0 : (2 * n_stars_i)],
                    ]
                )

                assert torch.all(
                    log_flux_mean[i, 0:n_stars_i, :].flatten()
                    == h_out[
                        i,
                        star_encoder.source_params_mean_indx_mat[n_stars_i][
                            0 : (n_bands * n_stars_i)
                        ],
                    ]
                )
                assert torch.all(
                    log_flux_logvar[i, 0:n_stars_i, :].flatten()
                    == h_out[
                        i,
                        star_encoder.source_params_var_indx_mat[n_stars_i][
                            0 : (n_bands * n_stars_i)
                        ],
                    ]
                )

        # test that everything works even when n_stars is None
        (
            loc_mean,
            loc_logvar,
            log_flux_mean,
            log_flux_logvar,
            log_probs,
        ) = star_encoder.forward(image_ptiles, n_sources=None)

        map_n_stars = torch.argmax(log_probs, dim=1)

        (
            _loc_mean,
            _loc_logvar,
            _log_flux_mean,
            _log_flux_logvar,
            _log_probs,
        ) = star_encoder.forward(image_ptiles, n_sources=map_n_stars)

        assert torch.all(loc_mean == _loc_mean)
        assert torch.all(loc_logvar == _loc_logvar)
        assert torch.all(log_flux_mean == _log_flux_mean)
        assert torch.all(log_flux_logvar == _log_flux_logvar)
        assert torch.all(log_probs == _log_probs)

    def test_forward_to_hidden2d(self):
        """
        * Consistency check of using forward vs get_var_params
        """

        n_image_tiles = 30
        max_detections = 4
        ptile_slen = 9
        n_bands = 2

        # get encoder
        star_encoder = sourcenet_lib.SourceEncoder(
            slen=101,
            ptile_slen=ptile_slen,
            step=2,
            edge_padding=3,
            n_bands=n_bands,
            max_detections=max_detections,
            n_source_params=n_bands,
        ).to(utils.device)

        star_encoder.eval()

        # simulate image padded tiles
        n_samples = 10
        image_ptiles = (
            torch.randn(n_image_tiles, n_bands, ptile_slen, ptile_slen) + 10.0
        ).to(utils.device)
        n_star_per_tile_sampled = (
            torch.Tensor(np.random.choice(max_detections, (n_samples, n_image_tiles)))
            .type(torch.LongTensor)
            .to(utils.device)
        )

        h = star_encoder._get_var_params_all(image_ptiles).detach()
        (
            loc_mean,
            loc_logvar,
            log_flux_mean,
            log_flux_logvar,
        ) = star_encoder._get_var_params_for_n_sources(h, n_star_per_tile_sampled)

        # CHECK THAT THIS MATCHES MY OLD PARAMETERS
        for i in range(n_samples):
            (
                loc_mean_i,
                loc_logvar_i,
                log_flux_mean_i,
                log_flux_logvar_i,
                _,
            ) = star_encoder.forward(image_ptiles, n_star_per_tile_sampled[i])

            assert (loc_mean_i - loc_mean[i]).abs().max() < 1e-6, (
                (loc_mean_i - loc_mean[i]).abs().max()
            )
            assert torch.all(loc_logvar_i == loc_logvar[i])
            assert torch.all(log_flux_mean_i == log_flux_mean[i])
            assert torch.all(log_flux_logvar_i == log_flux_logvar[i])

    def test_full_params_from_sampled(self):
        """
        * Check that we can recover full params from sampled params, and that they agree with the
        ones recovered from the tiles.
        """
        n_samples = 10
        ptile_slen = 9
        max_detections = 4
        n_bands = 2

        # get encoder
        star_encoder = sourcenet_lib.SourceEncoder(
            slen=101,
            ptile_slen=ptile_slen,
            step=2,
            edge_padding=3,
            n_bands=n_bands,
            max_detections=max_detections,
            n_source_params=n_bands,
        ).to(utils.device)

        n_image_tiles = star_encoder.tile_coords.shape[0]

        # draw sampled subimage parameters
        n_stars_sampled = (
            torch.Tensor(np.random.choice(max_detections, (n_samples, n_image_tiles)))
            .type(torch.long)
            .to(utils.device)
        )

        is_on_array = (
            utils.get_is_on_from_tile_n_sources_2d(n_stars_sampled, max_detections)
            .float()
            .to(utils.device)
        )

        subimage_locs_sampled = torch.rand(
            (n_samples, n_image_tiles, max_detections, 2)
        ).to(utils.device) * is_on_array.unsqueeze(3)

        subimage_fluxes_sampled = (
            torch.rand((n_samples, n_image_tiles, max_detections, n_bands)).to(
                utils.device
            )
            * is_on_array.unsqueeze(3)
        ).to(utils.device)

        (
            locs_full_image,
            fluxes_full_image,
            n_stars_full,
        ) = star_encoder._get_full_params_from_sampled_params(
            subimage_locs_sampled, subimage_fluxes_sampled, star_encoder.slen
        )

        # test against individually un-utiled parameters
        for i in range(n_samples):
            (
                locs_full_image_i,
                fluxes_full_image_i,
                n_stars_i,
            ) = celeste.utils.get_full_params_from_tile_params(
                subimage_locs_sampled[i],
                subimage_fluxes_sampled[i],
                star_encoder.tile_coords,
                star_encoder.slen,
                star_encoder.ptile_slen,
                star_encoder.edge_padding,
            )

            assert torch.all(locs_full_image_i == locs_full_image[i, 0:n_stars_i])
            assert torch.all(fluxes_full_image_i == fluxes_full_image[i, 0:n_stars_i])
