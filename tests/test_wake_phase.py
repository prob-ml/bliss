import torch
import numpy as np
import pytest

from celeste import device, use_cuda
from celeste import psf_transform
from celeste import wake


class TestStarWakeEncoder:
    def test_star_wake(self, trained_star_encoder, data_path, fitted_powerlaw_psf):
        # load the test image
        # 3-stars 30*30
        true_params = torch.load(data_path.joinpath("3star_test_params"))
        true_image = true_params["images"]

        # initialization
        ## initialize background params, which will create the true background
        init_background_params = torch.zeros(2, 3, device=device)
        init_background_params[0, 0] = 686.0
        init_background_params[1, 0] = 1123.0

        ## make sure test background equals the initialization
        init_background = wake.PlanarBackground(init_background_params, 30).forward()
        assert torch.all(init_background == true_params["background"].to(device))

        ## initialize psf params, just add 4 to each sigmas
        true_psf = fitted_powerlaw_psf
        psf_file = data_path.joinpath("fitted_powerlaw_psf_params.npy")
        true_psf_params = torch.from_numpy(np.load(psf_file)).to(device)

        init_psf_params = true_psf_params.clone()
        init_psf_params[0, 1:3] = init_psf_params[0, 1:3] + torch.tensor([4.0, 4.0]).to(
            device
        )
        init_psf_params[1, 1:3] = init_psf_params[1, 1:3] + torch.tensor([4.0, 4.0]).to(
            device
        )

        # get the trained encoder
        star_encoder = trained_star_encoder

        ## assert star_encoder is food enough
        locs, source_params, n_sources = trained_star_encoder.sample_encoder(
            true_image.to(device),
            n_samples=1,
            return_map_n_sources=True,
            return_map_source_params=True,
            training=False,
        )

        if not use_cuda:
            return

        assert n_sources == true_params["n_sources"].to(device)

        diff_locs = true_params["locs"].sort(1)[0].to(device) - locs.sort(1)[0]
        diff_locs *= true_image.size(-1)
        assert diff_locs.abs().max() <= 0.5

        # fluxes
        diff = (
            true_params["log_fluxes"].sort(1)[0].to(device) - source_params.sort(1)[0]
        )
        assert torch.all(diff.abs() <= source_params.sort(1)[0].abs() * 0.10)
        assert torch.all(
            diff.abs() <= true_params["log_fluxes"].sort(1)[0].abs().to(device) * 0.10
        )

        # run the wake-phase training
        estimate_params = wake.run_wake(
            true_image.to(device),
            star_encoder,
            init_psf_params,
            init_background_params,
            n_samples=100,
            out_filename="wake_estimate_params",
            n_epochs=2000,
            lr=1e-1,
            print_every=100,
            run_map=False,
        )

        # TODO: add assertions
        # propose: residuals PSF is less than 10% of the true PSF
        estimate_psf_params = list(estimate_params.power_law_psf.parameters())[0]
        estimate_psf = psf_transform.PowerLawPSF(estimate_psf_params).forward().detach()

        residuals = true_psf.to(device) - estimate_psf.to(device)
        assert torch.all(residuals.abs() <= true_psf.to(device).abs() * 0.1)
