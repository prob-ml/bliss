import torch
import numpy as np

from bliss.models import encoder


class TestSourceEncoder:
    def test_forward(self, device_setup):
        """
        * Test that forward returns the correct pattern of zeros.
        * Test that variational parameters inside h agree with those returned from forward.
        * Test everything works with n_stars=None in forward.
        """
        device = device_setup.device

        n_image_tiles = 30
        max_detections = 4
        ptile_slen = 8
        n_bands = 2
        tile_slen = 2

        # get encoder
        star_encoder = encoder.ImageEncoder(
            enc_conv_c=20,
            enc_hidden=256,
            slen=101,
            ptile_slen=ptile_slen,
            tile_slen=tile_slen,
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

            pred = star_encoder.forward(image_ptiles, n_star_per_tile)

            assert torch.all(0.0 <= pred["loc_mean"])
            assert torch.all(pred["loc_mean"] <= 1.0)

            # test we have the correct pattern of zeros
            assert ((pred["loc_mean"] != 0).sum(1)[:, 0] == n_star_per_tile).all()
            assert ((pred["loc_mean"] != 0).sum(1)[:, 1] == n_star_per_tile).all()

            assert ((pred["loc_logvar"] != 0).sum(1)[:, 0] == n_star_per_tile).all()
            assert ((pred["loc_logvar"] != 0).sum(1)[:, 1] == n_star_per_tile).all()

            for n in range(n_bands):
                assert (
                    (pred["loc_mean"][:, :, n] != 0).sum(1) == n_star_per_tile
                ).all()
                assert (
                    (pred["log_flux_mean"][:, :, n] != 0).sum(1) == n_star_per_tile
                ).all()

            # check pattern of zeros
            is_on_array = encoder.get_is_on_from_n_sources(
                n_star_per_tile, star_encoder.max_detections
            )
            _loc_mean = pred["loc_mean"] * is_on_array.unsqueeze(2).float()
            _log_flux_mean = pred["log_flux_mean"] * is_on_array.unsqueeze(2).float()
            assert torch.all(_loc_mean == pred["loc_mean"])
            assert torch.all(_log_flux_mean == pred["log_flux_mean"])

            # we check the variational parameters against the hidden parameters
            # one by one
            h_out = star_encoder._get_var_params_all(image_ptiles)

            # get index matrices
            locs_mean_indx_mat = star_encoder.indx_mats[0]
            locs_var_indx_mat = star_encoder.indx_mats[1]
            log_flux_mean_indx_mat = star_encoder.indx_mats[4]
            log_flux_var_indx_mat = star_encoder.indx_mats[5]
            prob_n_source_indx_mat = star_encoder.prob_n_source_indx

            for i in range(n_image_tiles):
                if n_star_per_tile[i] == 0:
                    assert torch.all(pred["loc_mean"][i] == 0)
                    assert torch.all(pred["loc_logvar"][i] == 0)
                    assert torch.all(pred["log_flux_mean"][i] == 0)
                    assert torch.all(pred["log_flux_logvar"][i] == 0)
                else:
                    n_stars_i = int(n_star_per_tile[i])

                    assert torch.all(
                        pred["loc_mean"][i, :n_stars_i].flatten()
                        == torch.sigmoid(h_out)[
                            i,
                            locs_mean_indx_mat[n_stars_i][: (2 * n_stars_i)],
                        ]
                    )

                    assert torch.all(
                        pred["loc_logvar"][i, :n_stars_i].flatten()
                        == h_out[
                            i,
                            locs_var_indx_mat[n_stars_i][: (2 * n_stars_i)],
                        ]
                    )

                    assert torch.all(
                        pred["log_flux_mean"][i, :n_stars_i].flatten()
                        == h_out[
                            i,
                            log_flux_mean_indx_mat[n_stars_i][: (n_bands * n_stars_i)],
                        ]
                    )

                    assert torch.all(
                        pred["log_flux_logvar"][i, :n_stars_i].flatten()
                        == h_out[
                            i,
                            log_flux_var_indx_mat[n_stars_i][: (n_bands * n_stars_i)],
                        ]
                    )

                    assert torch.all(
                        pred["n_source_log_probs"][i].flatten()
                        == star_encoder.log_softmax(h_out[:, prob_n_source_indx_mat])[i]
                    )

    def test_forward_to_hidden2d(self, device_setup):
        """Consistency check of using forward vs get_var_params"""
        device = device_setup.device

        n_image_tiles = 30
        max_detections = 4
        ptile_slen = 8
        tile_slen = 2
        n_bands = 2
        n_samples = 10

        # get encoder
        star_encoder = encoder.ImageEncoder(
            enc_conv_c=20,
            enc_hidden=256,
            slen=101,
            ptile_slen=ptile_slen,
            tile_slen=tile_slen,
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
            pred = star_encoder._get_var_params_for_n_sources(
                h, n_star_per_tile_sampled
            )

            #  test prediction matches tile by tile
            for i in range(n_samples):
                pred_i = star_encoder.forward(image_ptiles, n_star_per_tile_sampled[i])

                assert (pred_i["loc_mean"] - pred["loc_mean"][i]).abs().max() < 1e-6
                assert torch.all(pred_i["loc_logvar"] == pred["loc_logvar"][i])
                assert torch.all(pred_i["log_flux_mean"] == pred["log_flux_mean"][i])
                assert torch.all(
                    pred_i["log_flux_logvar"] == pred["log_flux_logvar"][i]
                )
                assert torch.all(
                    pred_i["galaxy_param_mean"] == pred["galaxy_param_mean"][i]
                )
                assert torch.all(
                    pred_i["galaxy_param_logvar"] == pred["galaxy_param_logvar"][i]
                )
