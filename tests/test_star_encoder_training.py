import numpy as np
import pytest
import torch

from celeste import device, use_cuda
from celeste import psf_transform
from celeste import train
from celeste import wake
from celeste.datasets import simulated_datasets
from celeste.models import sourcenet


class TestStarEncoderTraining:
    @pytest.fixture(scope="module")
    def trained_star_encoder(
        config_path, data_path, single_band_galaxy_decoder, fitted_powerlaw_psf
    ):
        # create training dataset
        n_bands = 2
        max_stars = 20
        mean_stars = 15
        min_stars = 5
        f_min = 1e4
        slen = 50

        # set background
        background = torch.zeros(n_bands, slen, slen, device=device)
        background[0] = 686.0
        background[1] = 1123.0

        # simulate dataset
        n_images = 128
        simulator_args = (
            single_band_galaxy_decoder,
            fitted_powerlaw_psf,
            background,
        )

        simulator_kwargs = dict(
            slen=slen,
            n_bands=n_bands,
            max_sources=max_stars,
            mean_sources=mean_stars,
            min_sources=min_stars,
            f_min=f_min,
            star_prob=1.0,  # enforce only stars are created in the training images.
        )

        dataset = simulated_datasets.SourceDataset(
            n_images, simulator_args, simulator_kwargs
        )

        # setup Star Encoder
        star_encoder = sourcenet.SourceEncoder(
            slen=slen,
            ptile_slen=8,
            step=2,
            edge_padding=3,
            n_bands=n_bands,
            max_detections=2,
            n_source_params=n_bands,  # star has n_bands # fluxes
            enc_conv_c=5,
            enc_kern=3,
            enc_hidden=64,
        ).to(device)

        # train encoder
        # training wrapper
        SleepTraining = train.SleepTraining(
            model=star_encoder,
            dataset=dataset,
            slen=slen,
            n_bands=n_bands,
            n_source_params=n_bands,  # star has n_bands # fluxes
            verbose=False,
            batchsize=32,
        )

        n_epochs = 100 if use_cuda else 1
        SleepTraining.run(n_epochs=n_epochs)

        return star_encoder

    @pytest.mark.parametrize("n_star", [1, 3])
    def test_star_sleep(self, trained_star_encoder, n_star, data_path):

        # load test image
        test_star = torch.load(data_path.joinpath(f"{n_star}star_test_params"))
        test_image = test_star["images"]

        assert test_star["fluxes"].min() > 0

        # get the estimated params
        locs, source_params, n_sources = trained_star_encoder.sample_encoder(
            test_image.to(device),
            n_samples=1,
            return_map_n_sources=True,
            return_map_source_params=True,
            training=False,
        )

        # we only expect our assert statements to be true
        # when the model is trained in full, which requires cuda
        if not use_cuda:
            return

        # test that parameters match.
        assert n_sources == test_star["n_sources"].to(device)

        diff_locs = test_star["locs"].sort(1)[0].to(device) - locs.sort(1)[0]
        diff_locs *= test_image.size(-1)
        assert diff_locs.abs().max() <= 0.5

        # fluxes
        diff = test_star["log_fluxes"].sort(1)[0].to(device) - source_params.sort(1)[0]
        assert torch.all(diff.abs() <= source_params.sort(1)[0].abs() * 0.10)
        assert torch.all(
            diff.abs() <= test_star["log_fluxes"].sort(1)[0].abs().to(device) * 0.10
        )

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
        init_psf_params[0, 1:3] += torch.tensor([4.0, 4.0]).to(device)
        init_psf_params[1, 1:3] += torch.tensor([4.0, 4.0]).to(device)

        # run the wake-phase training
        estimate_params = wake.run_wake(
            true_image.to(device),
            trained_star_encoder,
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

        if not use_cuda:
            return

        assert torch.all(residuals.abs() <= true_psf.to(device).abs() * 0.1)
