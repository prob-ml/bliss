import torch
import numpy as np
from itertools import permutations

from celeste import sleep, utils


class TestStarEncoderObjective:
    def test_get_all_comb_losses(self):
        # this checks that our function to return all combination of losses
        # is correct

        batchsize = 10
        max_detections = 4
        max_stars = 6
        n_bands = 2

        # true parameters
        true_locs = torch.rand(batchsize, max_stars, 2).to(utils.device)
        true_log_fluxes = torch.randn(batchsize, max_stars, n_bands).to(utils.device)

        # estimated parameters
        loc_mean = torch.randn(batchsize, max_detections, 2).to(utils.device)
        loc_log_var = torch.randn(batchsize, max_detections, 2).to(utils.device)

        log_flux_mean = torch.randn(batchsize, max_detections, n_bands).to(utils.device)
        log_flux_log_var = torch.randn(batchsize, max_detections, n_bands).to(
            utils.device
        )

        # get loss for locations
        locs_log_probs_all = sleep._get_locs_logprob_all_combs(
            true_locs, loc_mean, loc_log_var
        )

        # get loss for fluxes
        flux_log_probs_all = sleep._get_source_params_logprob_all_combs(
            true_log_fluxes, log_flux_mean, log_flux_log_var
        )

        # for my sanity
        assert list(locs_log_probs_all.shape) == [batchsize, max_stars, max_detections]
        assert list(flux_log_probs_all.shape) == [batchsize, max_stars, max_detections]

        for i in range(batchsize):
            for j in range(max_stars):
                for k in range(max_detections):
                    flux_loss_ij = utils.eval_normal_logprob(
                        true_log_fluxes[i, j],
                        log_flux_mean[i, k],
                        log_flux_log_var[i, k],
                    ).sum()

                    assert flux_loss_ij == flux_log_probs_all[i, j, k]

                    locs_loss_ij = utils.eval_normal_logprob(
                        true_locs[i, j], loc_mean[i, k], loc_log_var[i, k]
                    ).sum()

                    assert locs_loss_ij == locs_log_probs_all[i, j, k]

    def test_get_min_perm_loss(self):
        """
        Same as previous function but checks that we can get the permutation with the minimum loss.
        """

        batchsize = 100
        max_detections = 4
        max_stars = 4
        n_bands = 2

        # true parameters
        n_stars = torch.Tensor(np.random.choice(max_detections + 1, batchsize)).to(
            utils.device
        )
        is_on_array = utils.get_is_on_from_n_sources(n_stars, max_detections).float()

        true_locs = torch.rand(batchsize, max_detections, 2).to(
            utils.device
        ) * is_on_array.unsqueeze(2)
        true_log_fluxes = torch.randn(batchsize, max_detections, n_bands).to(
            utils.device
        ) * is_on_array.unsqueeze(2)

        # estimated parameters
        loc_mean = torch.randn(batchsize, max_detections, 2).to(
            utils.device
        ) * is_on_array.unsqueeze(2)
        loc_log_var = torch.randn(batchsize, max_detections, 2).to(
            utils.device
        ) * is_on_array.unsqueeze(2)

        log_flux_mean = torch.randn(batchsize, max_detections, n_bands).to(
            utils.device
        ) * is_on_array.unsqueeze(2)
        log_flux_log_var = torch.randn(batchsize, max_detections, n_bands).to(
            utils.device
        ) * is_on_array.unsqueeze(2)

        # get loss for locations
        locs_log_probs_all = sleep._get_locs_logprob_all_combs(
            true_locs, loc_mean, loc_log_var
        )

        # get loss for fluxes
        flux_log_probs_all = sleep._get_source_params_logprob_all_combs(
            true_log_fluxes, log_flux_mean, log_flux_log_var
        )

        locs_loss, fluxes_loss, _ = sleep._get_min_perm_loss(
            locs_log_probs_all, flux_log_probs_all, is_on_array
        )

        # a quick check for zer0 and one stars
        assert (locs_loss[n_stars == 0] == 0).all()
        assert (
            locs_loss[n_stars == 1] == -locs_log_probs_all[n_stars == 1][:, 0, 0]
        ).all()
        assert (
            fluxes_loss[n_stars == 1] == -flux_log_probs_all[n_stars == 1][:, 0, 0]
        ).all()

        # a more thorough check for all possible n_stars
        for i in range(batchsize):
            _n_stars = int(n_stars[i])

            if n_stars[i] == 0:
                assert locs_loss[i] == 0
                continue

            _true_locs = true_locs[i, 0:_n_stars, :]
            _loc_mean = loc_mean[i, 0:_n_stars, :]
            _loc_log_var = loc_log_var[i, 0:_n_stars, :]

            _true_log_fluxes = true_log_fluxes[i, 0:_n_stars, :]
            _log_flux_mean = log_flux_mean[i, 0:_n_stars, :]
            _log_flux_log_var = log_flux_log_var[i, 0:_n_stars, :]

            min_locs_loss = 1e16
            # min_log_fluxes_loss = 1e16
            for perm in permutations(range(_n_stars)):
                locs_loss_perm = -utils.eval_normal_logprob(
                    _true_locs, _loc_mean[perm, :], _loc_log_var[perm, :]
                )

                if locs_loss_perm.sum() < min_locs_loss:
                    min_locs_loss = locs_loss_perm.sum()
                    min_log_fluxes_loss = -utils.eval_normal_logprob(
                        _true_log_fluxes,
                        _log_flux_mean[perm, :],
                        _log_flux_log_var[perm, :],
                    ).sum()

            assert torch.abs(locs_loss[i] - min_locs_loss) < 1e-5, torch.abs(
                locs_loss[i] - min_locs_loss
            )
            assert torch.abs(fluxes_loss[i] - min_log_fluxes_loss) < 1e-5, torch.abs(
                fluxes_loss[i] - min_log_fluxes_loss
            )
