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
    def init_psf_setup(self, data_path):
        psf_file = data_path.joinpath("fitted_powerlaw_psf_params.npy")
        true_psf_params = torch.from_numpy(np.load(psf_file)).to(device)
        init_psf_params = true_psf_params.clone()[None, 0, ...]
        init_psf_params[0, 1:3] += torch.tensor([1.0, 1.0]).to(device)
        # init_psf_params[1, 1:3] += torch.tensor([1.0, 1.0]).to(device)

        init_psf = psf_transform.PowerLawPSF(init_psf_params).forward().detach()

        return {"init_psf_params": init_psf_params, "init_psf": init_psf}

    @pytest.fixture(scope="module")
    def trained_star_encoder(
        self, data_path, single_band_galaxy_decoder, init_psf_setup
    ):
        # create training dataset
        n_bands = 1
        max_stars = 20
        mean_stars = 10
        min_stars = 5
        f_min = 1e3
        slen = 50

        # set background for training
        background = torch.zeros(n_bands, slen, slen, device=device)
        background[0] = 686.0

        # set initialized psf
        init_psf = init_psf_setup["init_psf"]

        # simulate dataset
        n_images = 64 * 3
        simulator_args = (
            single_band_galaxy_decoder,
            init_psf,
            background,
        )

        simulator_kwargs = dict(
            slen=slen,
            n_bands=n_bands,
            max_sources=max_stars,
            mean_sources=mean_stars,
            min_sources=min_stars,
            f_min=f_min,
            prob_galaxy=0.0,  # enforce only stars are created in the training images.
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
            n_galaxy_params=8,  # star has n_bands # fluxes
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
            verbose=False,
            batchsize=32,
        )

        n_epochs = 300 if use_cuda else 1
        SleepTraining.run(n_epochs=n_epochs)

        return star_encoder

    def test_star_wake(
        self,
        trained_star_encoder,
        single_band_fitted_powerlaw_psf,
        init_psf_setup,
        test_star,
    ):
        # load the test image
        # 3-stars 30*30
        test_image = test_star["images"]

        # initialization
        # initialize background params, which will create the true background
        init_background_params = torch.zeros(1, 3, device=device)
        init_background_params[0, 0] = 686.0

        # make sure test background equals the initialization
        # init_background = (
        #     wake.PlanarBackground(init_background_params, 30).forward().detach()
        # )
        # assert torch.all(init_background == test_star["background"].to(device))

        # initialize psf params, just add 4 to each sigmas
        true_psf = single_band_fitted_powerlaw_psf.clone()
        init_psf_params = init_psf_setup["init_psf_params"]

        # run the wake-phase training
        n_epochs = 4000 if use_cuda else 1

        estimate_params, map_loss = wake.run_wake(
            test_image.to(device),
            trained_star_encoder,
            init_psf_params,
            init_background_params,
            n_samples=1000,
            n_epochs=n_epochs,
            lr=0.001,
            print_every=10,
            run_map=False,
        )

        # PSF residual reduce by 70%
        estimate_psf_params = list(estimate_params.power_law_psf.parameters())[0]
        estimate_psf = psf_transform.PowerLawPSF(estimate_psf_params).forward().detach()

        init_psf = init_psf_setup["init_psf"]
        init_residuals = true_psf.to(device) - init_psf.to(device)

        estimate_residuals = true_psf.to(device) - estimate_psf.to(device)

        if not use_cuda:
            return

        assert estimate_residuals.abs().sum() <= init_residuals.abs().sum() * 0.30
