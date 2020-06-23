import numpy as np
import pytest
import torch
import pytorch_lightning as ptl
from pytorch_lightning.profiler import AdvancedProfiler

from celeste import device, use_cuda
from celeste import psf_transform, wake, sleep
from celeste.models import encoder, decoder


class TestStarEncoderTraining:
    @pytest.fixture(scope="module")
    def init_psf_setup(self, data_path):
        psf_file = data_path.joinpath("fitted_powerlaw_psf_params.npy")
        true_psf_params = torch.from_numpy(np.load(psf_file)).to(device)
        init_psf_params = true_psf_params.clone()[None, 0, ...]
        init_psf_params[0, 1:3] += torch.tensor([1.0, 1.0]).to(device)

        init_psf = psf_transform.PowerLawPSF(init_psf_params).forward().detach()

        return {"init_psf_params": init_psf_params, "init_psf": init_psf}

    @pytest.fixture(scope="module")
    def trained_encoder(
        self,
        data_path,
        single_band_galaxy_decoder,
        single_band_fitted_powerlaw_psf,
        profile,
    ):
        # create training dataset
        n_bands = 1
        max_stars = 20
        mean_stars = 15
        min_stars = 5
        f_min = 1e4
        slen = 50

        # set background
        background = torch.zeros(n_bands, slen, slen, device=device)
        background.fill_(686.0)

        # simulate dataset
        n_images = 64 * 4
        simulator_args = (
            single_band_galaxy_decoder,
            single_band_fitted_powerlaw_psf,
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

        batch_size = 32
        n_batches = int(n_images / batch_size)
        dataset = decoder.SimulatedDataset(
            n_batches, batch_size, simulator_args, simulator_kwargs
        )

        # setup Star Encoder
        star_encoder = encoder.ImageEncoder(
            slen=slen,
            ptile_slen=8,
            step=2,
            edge_padding=3,
            n_bands=n_bands,
            max_detections=2,
            n_galaxy_params=single_band_galaxy_decoder.latent_dim,
            enc_conv_c=5,
            enc_kern=3,
            enc_hidden=64,
        ).to(device)

        # train encoder
        # training wrapper
        sleep_net = sleep.SleepPhase(dataset=dataset, image_encoder=star_encoder)

        n_epochs = 200 if use_cuda else 1
        # implement profiler
        profiler = AdvancedProfiler(output_filename=profile)

        # runs on gpu or cpu?
        n_device = [0] if use_cuda else 0  # 0 means no gpu

        sleep_trainer = ptl.Trainer(
            gpus=n_device,
            profiler=profiler,
            min_epochs=n_epochs,
            max_epochs=n_epochs,
            reload_dataloaders_every_epoch=True,
        )

        sleep_trainer.fit(sleep_net)

        return sleep_net.model

    def test_star_wake(
        self,
        trained_encoder,
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

        # initialize psf params, just add 4 to each sigmas
        true_psf = single_band_fitted_powerlaw_psf.clone()
        init_psf_params = init_psf_setup["init_psf_params"]

        n_samples = 1000 if use_cuda else 1
        hparams = {"n_samples": n_samples, "lr": 0.001}
        wake_phase_model = wake.WakePhase(
            trained_encoder,
            test_image,
            init_psf_params,
            init_background_params,
            hparams,
        )

        # run the wake-phase training
        n_epochs = 2800 if use_cuda else 1

        # implement tensorboard
        profiler = AdvancedProfiler(output_filename="wake_phase.txt")

        # runs on gpu or cpu?
        device_num = [0] if use_cuda else 0  # 0 means no gpu

        use_val = 10 if use_cuda else 1
        wake_trainer = ptl.Trainer(
            gpus=device_num,
            profiler=profiler,
            min_epochs=n_epochs,
            max_epochs=n_epochs,
            reload_dataloaders_every_epoch=True,
            check_val_every_n_epoch=use_val,
        )

        wake_trainer.fit(wake_phase_model)

        estimate_psf_params = list(
            wake_phase_model.model_params.power_law_psf.parameters()
        )[0]
        estimate_psf = psf_transform.PowerLawPSF(estimate_psf_params).forward().detach()

        init_psf = init_psf_setup["init_psf"]
        init_residuals = true_psf.to(device) - init_psf.to(device)

        estimate_residuals = true_psf.to(device) - estimate_psf.to(device)

        if not use_cuda:
            return

        assert estimate_residuals.abs().sum() <= init_residuals.abs().sum() * 0.30
