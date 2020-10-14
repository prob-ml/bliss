import pytest
import torch
import pytorch_lightning as pl
from bliss import wake


@pytest.fixture(scope="module")
def dataset(get_dataset, devices):
    batch_size = 128 if devices.use_cuda else 1
    n_batches = 10 if devices.use_cuda else 1
    overrides = ["model=basic_sleep_star"]
    dataset = get_dataset(batch_size, n_batches, overrides=overrides)
    return dataset


@pytest.fixture(scope="module")
def trained_encoder(dataset, get_trained_encoder, devices):
    n_epochs = 120 if devices.use_cuda else 1
    overrides = ["model=basic_sleep_star"]
    image_encoder = get_trained_encoder(n_epochs, dataset, overrides=overrides)
    return image_encoder


class TestStarSleepEncoder:
    @pytest.mark.parametrize("n_stars", ["1", "3"])
    def test_star_sleep(self, trained_encoder, n_stars, paths, devices):
        device = devices.device
        test_star = torch.load(paths["data"].joinpath(f"{n_stars}_star_test.pt"))
        test_image = test_star["images"].to(device)

        with torch.no_grad():
            # get the estimated params
            trained_encoder.eval()
            (
                n_sources,
                locs,
                galaxy_params,
                log_fluxes,
                galaxy_bool,
            ) = trained_encoder.map_estimate(test_image)

        # we only expect our assert statements to be true
        # when the model is trained in full, which requires cuda
        if not devices.use_cuda:
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
    def test_star_wake(self, trained_encoder, dataset, paths, devices):
        # load the test image
        # 3-stars 30*30 pixels.
        test_star = torch.load(paths["data"].joinpath("3_star_test.pt"))
        test_image = test_star["images"]
        test_slen = test_image.size(-1)

        # TODO: Reuse these when creating the background in the fixture
        # initialize background params, which will create the true background
        init_background_params = torch.zeros(1, 3, device=devices.device)
        init_background_params[0, 0] = 686.0

        n_samples = 1
        hparams = {"n_samples": n_samples, "lr": 0.001}
        image_decoder = dataset.image_decoder
        assert image_decoder.slen == test_slen
        wake_phase_model = wake.WakeNet(
            trained_encoder,
            image_decoder,
            test_image,
            init_background_params,
            hparams,
        )

        # run the wake-phase training
        n_epochs = 1

        wake_trainer = pl.Trainer(
            gpus=devices.gpus,
            profiler=None,
            logger=False,
            checkpoint_callback=False,
            min_epochs=n_epochs,
            max_epochs=n_epochs,
            reload_dataloaders_every_epoch=True,
        )

        wake_trainer.fit(wake_phase_model)
