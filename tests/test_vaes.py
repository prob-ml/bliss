import torch
import numpy as np

from celeste import device
from celeste.models import encoder


class TestSourceEncoder:
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
        star_encoder = encoder.ImageEncoder(
            slen=101,
            ptile_slen=ptile_slen,
            step=2,
            edge_padding=3,
            n_bands=n_bands,
            max_detections=max_detections,
            n_galaxy_params=8,
        ).to(device)

        with torch.no_grad():
            star_encoder.eval()

            # simulate image padded tiles
            image_ptiles = (
                torch.randn(n_image_tiles, n_bands, ptile_slen, ptile_slen) + 10.0
            ).to(device)

            n_star_per_tile = (
                torch.from_numpy(np.random.choice(max_detections, n_image_tiles))
                .type(torch.LongTensor)
                .to(device)
            )

            # forward
            (
                n_source_log_probs,
                loc_mean,
                loc_logvar,
                galaxy_param_mean,
                galaxy_param_logvar,
                log_flux_mean,
                log_flux_logvar,
                prob_galaxy,
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
            is_on_array = encoder.get_is_on_from_n_sources(
                n_star_per_tile, star_encoder.max_detections
            )
            assert torch.all((loc_mean * is_on_array.unsqueeze(2).float()) == loc_mean)
            assert torch.all(
                (loc_logvar * is_on_array.unsqueeze(2).float()) == loc_logvar
            )

            assert torch.all(
                (log_flux_mean * is_on_array.unsqueeze(2).float()) == log_flux_mean
            )
            assert torch.all(
                (log_flux_logvar * is_on_array.unsqueeze(2).float()) == log_flux_logvar
            )

            # we check the variational parameters against the hidden parameters
            # one by one
            h_out = star_encoder._get_var_params_all(image_ptiles)

            # get index matrices
            locs_mean_indx_mat = star_encoder.indx_mats[0]
            locs_var_indx_mat = star_encoder.indx_mats[1]
            log_flux_mean_indx_mat = star_encoder.indx_mats[4]
            log_flux_var_indx_mat = star_encoder.indx_mats[5]
            prob_galaxy_indx_mat = star_encoder.indx_mats[6]
            prob_n_source_indx_mat = star_encoder.prob_n_source_indx

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
                            i, locs_mean_indx_mat[n_stars_i][0 : (2 * n_stars_i)],
                        ]
                    )
                    assert torch.all(
                        loc_logvar[i, 0:n_stars_i, :].flatten()
                        == h_out[i, locs_var_indx_mat[n_stars_i][0 : (2 * n_stars_i)],]
                    )

                    assert torch.all(
                        log_flux_mean[i, 0:n_stars_i, :].flatten()
                        == h_out[
                            i,
                            log_flux_mean_indx_mat[n_stars_i][
                                0 : (n_bands * n_stars_i)
                            ],
                        ]
                    )
                    assert torch.all(
                        log_flux_logvar[i, 0:n_stars_i, :].flatten()
                        == h_out[
                            i,
                            log_flux_var_indx_mat[n_stars_i][0 : (n_bands * n_stars_i)],
                        ]
                    )

                    assert torch.all(
                        n_source_log_probs[i, :].flatten()
                        == star_encoder.log_softmax(h_out[:, prob_n_source_indx_mat])[i]
                    )

                    assert torch.all(
                        prob_galaxy[i, 0:n_stars_i, :].flatten()
                        == torch.sigmoid(h_out)[
                            i, prob_galaxy_indx_mat[n_stars_i][0 : (1 * n_stars_i)]
                        ]
                    )

    def test_forward_to_hidden2d(self):
        """
        * Consistency check of using forward vs get_var_params
        """

        n_image_tiles = 30
        max_detections = 4
        ptile_slen = 9
        n_bands = 2
        n_samples = 10

        # get encoder
        star_encoder = encoder.ImageEncoder(
            slen=101,
            ptile_slen=ptile_slen,
            step=2,
            edge_padding=3,
            n_bands=n_bands,
            max_detections=max_detections,
            n_galaxy_params=8,
        ).to(device)

        with torch.no_grad():
            star_encoder.eval()

            # simulate image padded tiles
            image_ptiles = (
                torch.randn(
                    n_image_tiles, n_bands, ptile_slen, ptile_slen, device=device
                )
                + 10.0
            )
            n_star_per_tile_sampled = torch.from_numpy(
                np.random.choice(max_detections, (n_samples, n_image_tiles))
            )

            h = star_encoder._get_var_params_all(image_ptiles).detach()
            (
                loc_mean,
                loc_logvar,
                galaxy_param_mean,
                galaxy_param_logvar,
                log_flux_mean,
                log_flux_logvar,
                prob_galaxy,
            ) = star_encoder._get_var_params_for_n_sources(h, n_star_per_tile_sampled)

            #  test prediction matches tile by tile
            for i in range(n_samples):
                (
                    _,
                    loc_mean_i,
                    loc_logvar_i,
                    galaxy_param_mean_i,
                    galaxy_param_logvar_i,
                    log_flux_mean_i,
                    log_flux_logvar_i,
                    prob_galaxy_i,
                ) = star_encoder.forward(image_ptiles, n_star_per_tile_sampled[i])

                assert (loc_mean_i - loc_mean[i]).abs().max() < 1e-6
                assert torch.all(loc_logvar_i == loc_logvar[i])
                assert torch.all(log_flux_mean_i == log_flux_mean[i])
                assert torch.all(log_flux_logvar_i == log_flux_logvar[i])

                assert torch.all(galaxy_param_mean_i == galaxy_param_mean[i])
                assert torch.all(galaxy_param_logvar_i == galaxy_param_logvar[i])
