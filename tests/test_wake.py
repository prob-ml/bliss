import numpy as np
import pytest
import torch
import pytorch_lightning as ptl
from pytorch_lightning.profiler import AdvancedProfiler

from celeste import use_cuda, psf_transform, wake


class TestStarEncoderTraining:
    @pytest.fixture(scope="module")
    def init_psf_setup(self, data_path, device):
        psf_file = data_path.joinpath("fitted_powerlaw_psf_params.npy")
        true_psf_params = torch.from_numpy(np.load(psf_file)).to(device)
        init_psf_params = true_psf_params.clone()[None, 0, ...]
        init_psf_params[0, 1:3] += torch.tensor([1.0, 1.0]).to(device)

        init_psf = psf_transform.PowerLawPSF(init_psf_params).forward().detach()

        return {"init_psf_params": init_psf_params, "init_psf": init_psf}

    @pytest.fixture(scope="module")
    def trained_encoder(
        self,
        get_trained_encoder,
        single_band_galaxy_decoder,
        init_psf_setup,
        device,
        device_id,
        profile,
    ):
        return get_trained_encoder(
            single_band_galaxy_decoder,
            init_psf_setup["init_psf"],
            device,
            device_id,
            profile,
            n_images=64 * 6,
            batch_size=32,
            n_epochs=200,
        )

    def test_star_wake(
        self,
        trained_encoder,
        single_band_fitted_powerlaw_psf,
        init_psf_setup,
        test_star,
        device,
        device_id,
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
        device_num = [device_id] if use_cuda else 0  # 0 means no gpu

        wake_trainer = ptl.Trainer(
            gpus=device_num,
            profiler=profiler,
            min_epochs=n_epochs,
            max_epochs=n_epochs,
            reload_dataloaders_every_epoch=True,
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
