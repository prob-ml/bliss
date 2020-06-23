import torch
import pytest
from celeste import use_cuda


@pytest.fixture(scope="module")
def trained_encoder(
    get_trained_encoder,
    single_band_galaxy_decoder,
    single_band_fitted_powerlaw_psf,
    device,
    device_id,
    profile,
):
    return get_trained_encoder(
        single_band_galaxy_decoder,
        single_band_fitted_powerlaw_psf,
        device,
        device_id,
        profile,
    )


class TestStarSleepEncoder:
    def test_star_sleep(self, trained_encoder, test_star, device):

        # load test image
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
