import pytest
import torch


class TestSleepStar:
    @pytest.fixture(scope="class")
    def trained_encoder(self, devices, get_trained_encoder):
        use_cuda = devices.use_cuda
        overrides = dict(
            model="basic_sleep_star",
            dataset="default" if use_cuda else "cpu",
            training="tests_default" if use_cuda else "cpu",
        )
        return get_trained_encoder(overrides)

    def test_star_sleep(self, n_stars, trained_encoder, paths, devices):
        device = devices.device
        test_path = paths["data"].joinpath(f"{n_stars}_star_test.pt")
        test_star = torch.load(test_path, map_location="cpu")
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

        # test n_sources
        true_n_sources = test_star["n_sources"].int().item()
        n_sources = n_sources.int().item()
        assert n_sources == true_n_sources

        # test locs.
        true_locs = test_star["locs"].reshape(-1, 2).to(device).sort(1)[0]
        true_locs = true_locs[:true_n_sources]
        locs = locs.reshape(-1, 2).to(device).sort(1)[0]
        locs = locs[:n_sources]

        diff_locs = true_locs - locs
        diff_locs *= test_image.size(-1)
        assert diff_locs.abs().max() <= 0.5

        # test fluxes
        n_bands = log_fluxes.shape[-1]
        true_log_fluxes = test_star["log_fluxes"].reshape(-1, n_bands)
        true_log_fluxes = true_log_fluxes.to(device).sort(1)[0][:true_n_sources]
        log_fluxes = log_fluxes.reshape(-1, n_bands).to(device).sort(1)[0][:n_sources]
        diff = true_log_fluxes - log_fluxes
        assert torch.all(diff.abs() <= log_fluxes * 0.1)
        assert torch.all(diff.abs() <= true_log_fluxes * 0.1)
