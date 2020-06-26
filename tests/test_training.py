import numpy as np
import pytest
import torch
import pytorch_lightning as pl
from pytorch_lightning.profiler import AdvancedProfiler

from celeste import use_cuda, psf_transform, wake, sleep
from celeste.models import decoder, encoder


def get_trained_encoder(
    galaxy_decoder,
    psf,
    device,
    device_id,
    profile,
    save_logs,
    n_bands=1,
    max_stars=20,
    mean_stars=15,
    min_stars=5,
    f_min=1e4,
    slen=50,
    n_images=128,
    batch_size=32,
    n_epochs=200,
    prob_galaxy=0.0,
):
    assert galaxy_decoder.n_bands == psf.size(0) == n_bands

    n_epochs = n_epochs if use_cuda else 1

    background = torch.zeros(n_bands, slen, slen, device=device)
    background.fill_(686.0)

    simulator_args = (
        galaxy_decoder,
        psf,
        background,
    )

    simulator_kwargs = dict(
        slen=slen,
        n_bands=n_bands,
        max_sources=max_stars,
        mean_sources=mean_stars,
        min_sources=min_stars,
        f_min=f_min,
        prob_galaxy=prob_galaxy,
    )

    n_batches = int(n_images / batch_size)
    dataset = decoder.SimulatedDataset(
        n_batches, batch_size, simulator_args, simulator_kwargs
    )

    # setup Star Encoder
    image_encoder = encoder.ImageEncoder(
        slen=slen,
        ptile_slen=8,
        step=2,
        edge_padding=3,
        n_bands=n_bands,
        max_detections=2,
        n_galaxy_params=galaxy_decoder.latent_dim,
        enc_conv_c=5,
        enc_kern=3,
        enc_hidden=64,
    ).to(device)

    # training wrapper
    sleep_net = sleep.SleepPhase(
        dataset=dataset, image_encoder=image_encoder, save_logs=save_logs
    )

    profiler = AdvancedProfiler(output_filename=profile) if profile != None else None

    # runs on gpu or cpu?
    n_device = [device_id] if use_cuda else 0  # 0 means no gpu

    sleep_trainer = pl.Trainer(
        logger=save_logs,
        checkpoint_callback=save_logs,
        gpus=n_device,
        profiler=profiler,
        min_epochs=n_epochs,
        max_epochs=n_epochs,
        reload_dataloaders_every_epoch=True,
    )

    sleep_trainer.fit(sleep_net)

    return sleep_net.image_encoder


class TestStarSleepEncoder:
    @pytest.fixture(scope="class")
    def trained_encoder(
        self,
        single_band_galaxy_decoder,
        single_band_fitted_powerlaw_psf,
        device,
        device_id,
        sprof,
        log,
    ):
        return get_trained_encoder(
            single_band_galaxy_decoder,
            single_band_fitted_powerlaw_psf,
            device,
            device_id,
            sprof,
            log,
            prob_galaxy=0.0,  # only stars will be drawn.
        )

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


class TestStarEncoderTraining:
    @pytest.fixture(scope="module")
    def init_psf_setup(self, data_path, device):
        psf_file = data_path.joinpath("fitted_powerlaw_psf_params.npy")
        true_psf_params = torch.from_numpy(np.load(psf_file)).to(device)
        init_psf_params = true_psf_params.clone()[None, 0, ...]
        init_psf_params[0, 1:3] += torch.tensor([1.0, 1.0]).to(device)

        init_psf = psf_transform.PowerLawPSF(init_psf_params).forward().detach()

        return {"init_psf_params": init_psf_params, "init_psf": init_psf}

    @pytest.fixture(scope="class")
    def trained_encoder(
        self, single_band_galaxy_decoder, init_psf_setup, device, device_id, sprof, log
    ):
        return get_trained_encoder(
            single_band_galaxy_decoder,
            init_psf_setup["init_psf"],
            device,
            device_id,
            sprof,
            log,
            n_images=64 * 6,
            batch_size=32,
            n_epochs=200,
            prob_galaxy=0.0,
        )

    def test_star_wake(
        self,
        trained_encoder,
        single_band_fitted_powerlaw_psf,
        init_psf_setup,
        test_3_stars,
        device,
        device_id,
        wprof,
        log,
    ):
        # load the test image
        # 3-stars 30*30
        test_image = test_3_stars["images"]

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
            log,
        )

        # run the wake-phase training
        n_epochs = 2800 if use_cuda else 1

        # implement tensorboard
        profiler = AdvancedProfiler(output_filename=wprof) if wprof != None else None

        # runs on gpu or cpu?
        device_num = [device_id] if use_cuda else 0  # 0 means no gpu

        wake_trainer = pl.Trainer(
            logger=log,
            checkpoint_callback=log,
            gpus=device_num,
            profiler=profiler,
            min_epochs=n_epochs,
            max_epochs=n_epochs,
            reload_dataloaders_every_epoch=True,
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
