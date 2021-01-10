from itertools import permutations

import numpy as np
import torch
from torch.distributions import Normal

from bliss.sleep import _get_params_logprob_all_combs, _get_min_perm_loss
from bliss.models.encoder import get_is_on_from_n_sources


class TestStarEncoderObjective:
    def test_get_params_logprob_all_combs(self, devices):
        # this checks that our function to return all combination of losses
        # is correct
        device = devices.device

        n_ptiles = 10
        max_detections = 4
        n_source_params = 8

        # true parameters
        true_params = torch.rand(
            n_ptiles, max_detections, n_source_params, device=device
        )

        # estimated parameters
        param_mean = torch.randn(
            n_ptiles, max_detections, n_source_params, device=device
        )
        param_logvar = torch.randn(
            n_ptiles, max_detections, n_source_params, device=device
        )

        # get all losses
        param_log_probs_all = _get_params_logprob_all_combs(
            true_params, param_mean, param_logvar
        )

        # just for my sanity
        assert list(param_log_probs_all.shape) == [
            n_ptiles,
            max_detections,
            max_detections,
        ]

        # for each batch, check that the ith estimated parameter is correctly
        # matched with the jth true parameter
        for n in range(n_ptiles):
            for i in range(max_detections):
                for j in range(max_detections):

                    param_loglik_ij = (
                        Normal(
                            param_mean[n, i],
                            (torch.exp(param_logvar[n, i]) + 1e-5).sqrt(),
                        )
                        .log_prob(true_params[n, j])
                        .sum()
                    )

                    assert param_loglik_ij == param_log_probs_all[n, i, j]

    def test_get_min_perm_loss(self, devices):
        """
        Same as previous function but checks that we can get the permutation with the minimum loss.
        """

        device = devices.device

        # data parameters
        n_ptiles = 100
        n_bands = 2
        max_detections = 4

        # true parameters
        true_n_sources = torch.from_numpy(
            np.random.choice(max_detections + 1, n_ptiles)
        ).to(device)
        true_is_on_array = get_is_on_from_n_sources(
            true_n_sources, max_detections
        ).float()

        # locations, fluxes and galaxy parameters
        true_locs = torch.rand(
            n_ptiles, max_detections, 2, device=device
        ) * true_is_on_array.unsqueeze(2)
        true_log_fluxes = torch.randn(
            n_ptiles, max_detections, n_bands, device=device
        ) * true_is_on_array.unsqueeze(2)

        # boolean indicating whether source is galaxy
        true_galaxy_bool = (
            (torch.rand(n_ptiles, max_detections) > 0.5).float().to(device)
        )

        # estimated parameters
        loc_mean = torch.randn(
            n_ptiles, max_detections, 2, device=device
        ) * true_is_on_array.unsqueeze(2)
        loc_mean = loc_mean + (true_is_on_array == 0).float().unsqueeze(-1) * 1e16
        loc_logvar = torch.randn(
            n_ptiles, max_detections, 2, device=device
        ) * true_is_on_array.unsqueeze(2)

        log_flux_mean = torch.randn(
            n_ptiles, max_detections, n_bands, device=device
        ) * true_is_on_array.unsqueeze(2)
        log_flux_logvar = torch.randn(
            n_ptiles, max_detections, n_bands, device=device
        ) * true_is_on_array.unsqueeze(2)

        # for each detection, prob that it is a galaxy
        prob_galaxy = torch.rand(n_ptiles, max_detections, device=device)

        # get loss for locations
        locs_log_probs_all = _get_params_logprob_all_combs(
            true_locs, loc_mean, loc_logvar
        )

        # get loss for fluxes
        star_params_log_probs_all = _get_params_logprob_all_combs(
            true_log_fluxes, log_flux_mean, log_flux_logvar
        )

        (locs_loss, star_params_loss, galaxy_bool_loss,) = _get_min_perm_loss(
            locs_log_probs_all,
            star_params_log_probs_all,
            prob_galaxy,
            true_galaxy_bool,
            true_is_on_array,
        )

        # when no sources, all losses should be zero
        assert (locs_loss[true_n_sources == 0] == 0).all()
        assert (star_params_loss[true_n_sources == 0] == 0).all()
        assert (galaxy_bool_loss[true_n_sources == 0] == 0).all()
        assert (
            locs_loss[true_n_sources == 1]
            == -locs_log_probs_all[true_n_sources == 1][:, 0, 0]
        ).all()

        # when there are no stars: stars loss should be zero
        which_no_stars = ((1 - true_galaxy_bool) * true_is_on_array).sum(1) == 0
        assert (star_params_loss[which_no_stars] == 0).all()

        # when there is only one source, and that source is a star
        which_one_star = (true_n_sources == 1) & (true_galaxy_bool[:, 0] == 0)
        assert (
            star_params_loss[which_one_star]
            == -star_params_log_probs_all[which_one_star][:, 0, 0]
        ).all()

        # a more thorough check for all possible true_n_sources
        for i in range(n_ptiles):
            _true_n_sources = int(true_n_sources[i])
            _true_galaxy_bool = true_galaxy_bool[i, 0:_true_n_sources]

            if true_n_sources[i] == 0:
                assert locs_loss[i] == 0
                assert star_params_loss[i] == 0
                assert galaxy_bool_loss[i] == 0
                continue

            # get parameters for ith observation
            _true_locs = true_locs[i, 0:_true_n_sources, :]
            _loc_mean = loc_mean[i, 0:_true_n_sources, :]
            _loc_logvar = loc_logvar[i, 0:_true_n_sources, :]

            _true_log_fluxes = true_log_fluxes[i, 0:_true_n_sources, :]
            _log_flux_mean = log_flux_mean[i, 0:_true_n_sources, :]
            _log_flux_logvar = log_flux_logvar[i, 0:_true_n_sources, :]

            _prob_galaxy = prob_galaxy[i, 0:_true_n_sources]

            min_locs_loss = 1e16
            min_star_params_loss = 1e16
            min_galaxy_bool_loss = 1e16
            for perm in permutations(range(_true_n_sources)):
                locs_loss_perm = -Normal(
                    _loc_mean[perm, :], (torch.exp(_loc_logvar[perm, :]) + 1e-5).sqrt()
                ).log_prob(_true_locs)

                star_params_loss_perm = (
                    -Normal(
                        _log_flux_mean[perm, :],
                        (torch.exp(_log_flux_logvar[perm, :]) + 1e-5).sqrt(),
                    )
                    .log_prob(_true_log_fluxes)
                    .sum(-1)
                )

                p = _prob_galaxy.unsqueeze(0)[:, perm].squeeze()
                galaxy_bool_loss_perm = -_true_galaxy_bool * torch.log(p)
                galaxy_bool_loss_perm -= (1 - _true_galaxy_bool) * torch.log(1 - p)

                if locs_loss_perm.sum() < min_locs_loss:
                    min_locs_loss = locs_loss_perm.sum()
                    min_star_params_loss = (
                        star_params_loss_perm * (1 - _true_galaxy_bool)
                    ).sum()
                    min_galaxy_bool_loss = galaxy_bool_loss_perm.sum()

            assert torch.abs(locs_loss[i] - min_locs_loss) < 1e-5
            assert torch.abs(star_params_loss[i] - min_star_params_loss) < 1e-5
            assert torch.abs(galaxy_bool_loss[i] - min_galaxy_bool_loss) < 1e-5
