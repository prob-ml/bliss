import pytest
import torch
import pytorch_lightning as pl

from bliss import use_cuda, wake
from bliss.models.decoder import get_mgrid, PowerLawPSF


class TestStarSleepEncoder:
    @pytest.fixture(scope="class")
    def trained_encoder(
        self, init_psf_setup, get_star_dataset, get_trained_encoder,
    ):
        star_dataset = get_star_dataset(
            fitted_psf_params, n_bands=1, slen=50, batch_size=32
        )
        trained_encoder = get_trained_encoder(star_dataset, n_epochs=100)
        return trained_encoder

    @pytest.mark.parametrize("n_stars", ["1", "3"])
    def test_star_sleep(self, trained_encoder, n_stars, data_path, device):
        test_star = torch.load(data_path.joinpath(f"{n_stars}_star_test.pt"))
        test_image = test_star["images"]

        trained_encoder = trained_encoder.to(device)

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
                test_image,
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


class TestStarWakeNet:
    @pytest.fixture(scope="class")
    def init_psf_setup(self, fitted_psf_params, device):
        # initialize psf params, just add 1 to each sigmas
        init_psf_params = fitted_psf_params.clone()[None, 0]
        init_psf_params[0, 1:3] += torch.tensor([1.0, 1.0]).to(device)
        init_psf = PowerLawPSF(init_psf_params).forward().detach()

        return {"init_psf_params": init_psf_params, "init_psf": init_psf}

    def test_star_wake(
        self,
        get_trained_encoder,
        get_star_dataset,
        fitted_psf_params,
        init_psf_setup,
        test_3_stars,
        device,
        gpus,
    ):
        # get dataset and encoder
        star_dataset = get_star_dataset(
            init_psf_setup["init_psf_params"],
            n_bands=1,
            slen=50,
            batch_size=32,
            n_images=64 * 6 if use_cuda else 32,
        )
        n_epochs = 200 if use_cuda else 1
        trained_encoder = get_trained_encoder(star_dataset, n_epochs=n_epochs)

        # load the test image
        # 3-stars 30*30 pixels.
        test_image = test_3_stars["images"]
        test_slen = test_image.size(-1)

        # TODO: Reuse these when creating the background in the fixture
        # initialize background params, which will create the true background
        init_background_params = torch.zeros(1, 3, device=device)
        init_background_params[0, 0] = 686.0

        n_samples = 1000 if use_cuda else 1
        hparams = {"n_samples": n_samples, "lr": 0.001}
        image_decoder = star_dataset.image_decoder
        image_decoder.slen = test_slen
        image_decoder.cached_grid = get_mgrid(test_slen)
        wake_phase_model = wake.WakeNet(
            trained_encoder,
            image_decoder,
            test_image,
            init_background_params,
            hparams,
        )

        # run the wake-phase training
        n_epochs = 2800 if use_cuda else 1

        wake_trainer = pl.Trainer(
            gpus=gpus,
            profiler=None,
            logger=False,
            checkpoint_callback=False,
            min_epochs=n_epochs,
            max_epochs=n_epochs,
            reload_dataloaders_every_epoch=True,
        )

        wake_trainer.fit(wake_phase_model)

        # round up psfs
        init_psf = init_psf_setup["init_psf"]
        true_psf = PowerLawPSF(fitted_psf_params).forward()[None, 0]
        estimated_psf_params = list(
            wake_phase_model.image_decoder.power_law_psf.parameters()
        )[0]
        estimated_psf = PowerLawPSF(estimated_psf_params).forward().detach()

        init_residuals = true_psf.to(device) - init_psf.to(device)
        estimated_residuals = true_psf.to(device) - estimated_psf.to(device)

        if not use_cuda:
            return

        assert estimated_residuals.abs().sum() <= init_residuals.abs().sum() * 0.30
