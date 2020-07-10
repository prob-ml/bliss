import numpy as np
import pytest
import torch
import pytorch_lightning as pl

from bliss import use_cuda, psf_transform, wake


class TestStarSleepEncoder:
    @pytest.fixture(scope="class")
    def trained_encoder(
        self, fitted_psf, get_star_dataset, get_trained_star_encoder,
    ):
        star_dataset = get_star_dataset(fitted_psf, n_bands=1, slen=50, batch_size=32)
        trained_encoder = get_trained_star_encoder(star_dataset, n_epochs=1)
        return trained_encoder

    @pytest.mark.only
    @pytest.mark.parametrize("n_stars", ["1", "3"])
    def test_star_sleep(self, trained_encoder, n_stars, data_path, device):
        test_star = torch.load(data_path.joinpath(f"{n_stars}_star_test.pt"))
        test_image = test_star["images"]

        with torch.no_grad():
            # get the estimated params
            trained_encoder.eval()
            (
                n_sources,
                locs,
                galaxy_params,
                log_fluxes,
                galaxy_bool,
            ) = trained_encoder.sample_encoder(
                test_image.to(device),
                n_samples=1,
                return_map_n_sources=True,
                return_map_source_params=True,
            )

        # we only expect our assert statements to be true
        # when the model is trained in full, which requires cuda
        if not use_cuda:
            return

        # test n_sources and locs
        assert n_sources == test_star["n_sources"].to(device)

        diff_locs = test_star["locs"].sort(1)[0].to(device) - locs.sort(1)[0]
        diff_locs *= test_image.size(-1)
        assert diff_locs.abs().max() <= 0.5

        # test fluxes
        diff = test_star["log_fluxes"].sort(1)[0].to(device) - log_fluxes.sort(1)[0]
        assert torch.all(diff.abs() <= log_fluxes.sort(1)[0].abs() * 0.10)
        assert torch.all(
            diff.abs() <= test_star["log_fluxes"].sort(1)[0].abs().to(device) * 0.10
        )


class TestStarWakePhase:
    @pytest.fixture(scope="class")
    def init_psf_setup(self, data_path, device):
        psf_file = data_path.joinpath("fitted_powerlaw_psf_params.npy")
        true_psf_params = torch.from_numpy(np.load(psf_file)).to(device)
        init_psf_params = true_psf_params.clone()[None, 0, ...]
        init_psf_params[0, 1:3] += torch.tensor([1.0, 1.0]).to(device)

        init_psf = psf_transform.PowerLawPSF(init_psf_params).forward().detach()

        return {"init_psf_params": init_psf_params, "init_psf": init_psf}

    @pytest.fixture(scope="class")
    def trained_encoder(
        self,
        star_dataset,
        init_psf_setup,
        device,
        device_id,
        profiler,
        save_logs,
        logs_path,
    ):
        star_dataset.image_decoder.psf = init_psf_setup["init_psf"]
        return get_trained_encoder(
            star_dataset, device, device_id, profiler, save_logs, logs_path,
        )

    def test_star_wake(
        self,
        trained_encoder,
        fitted_psf,
        init_psf_setup,
        test_3_stars,
        device,
        device_id,
        profiler,
        save_logs,
        logs_path,
    ):
        # load the test image
        # 3-stars 30*30
        test_image = test_3_stars["images"]
        star_dataset = get_star_dataset(
            fitted_psf, data_path, device, slen=test_image.size(-1)
        )

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
            save_logs,
        )

        # run the wake-phase training
        n_epochs = 2800 if use_cuda else 1

        # runs on gpu or cpu?
        device_num = [device_id] if use_cuda else 0  # 0 means no gpu

        wake_trainer = pl.Trainer(
            gpus=device_num,
            profiler=profiler,
            min_epochs=n_epochs,
            max_epochs=n_epochs,
            reload_dataloaders_every_epoch=True,
            default_root_dir=logs_path,
        )

        wake_trainer.fit(wake_phase_model)

        estimate_psf_params = list(wake_phase_model.power_law_psf.parameters())[0]
        estimate_psf = psf_transform.PowerLawPSF(estimate_psf_params).forward().detach()

        init_psf = init_psf_setup["init_psf"]
        init_residuals = true_psf.to(device) - init_psf.to(device)

        estimate_residuals = true_psf.to(device) - estimate_psf.to(device)

        if not use_cuda:
            return

        assert estimate_residuals.abs().sum() <= init_residuals.abs().sum() * 0.30
