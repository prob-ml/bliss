import pytest
import torch


class TestGalaxyEncoder:
    @pytest.fixture(scope="class")
    def trained_encoder(self, decoder_setup, encoder_setup, device_setup):
        use_cuda = device_setup.use_cuda

        # simulates either 1 or 2 galaxies in a 50 x 50 image
        # the input to the encoder is the 50 x 50 image
        # the encoder looks at 8x8 padded tile, and detects galaxies in 2x2 tile.
        slen = 50
        tile_slen = 2
        galaxy_dataset = decoder_setup.get_binary_dataset(
            slen=slen,
            tile_slen=tile_slen,
            batch_size=32 if use_cuda else 2,
            n_batches=10 if use_cuda else 2,
            max_sources_per_tile=2,
            min_sources_per_tile=1,
            loc_max_per_tile=0.8,
            loc_min_per_tile=0.2,
            # this is so that prob(n_source = 1) \approx prob(n_source = 2) \approx = 0.5
            # under the poisson prior
            mean_sources_per_tile=1.67,
            prob_galaxy=1.0,
            ptile_padding=0,
        )
        trained_encoder = encoder_setup.get_trained_encoder(
            galaxy_dataset,
            n_epochs=120 if use_cuda else 1,
            max_detections=2,
            ptile_slen=8,
            tile_slen=tile_slen,
            enc_conv_c=5,
            enc_kern=3,
            enc_hidden=64,
            validation_plot_start=1000,
        )
        return trained_encoder.to(device_setup.device)

    @pytest.mark.parametrize("n_galaxies", ["2"])
    def test_n_sources_and_locs(self, trained_encoder, n_galaxies, paths, device_setup):
        use_cuda = device_setup.use_cuda
        device = device_setup.device

        test_galaxy = torch.load(paths["data"].joinpath(f"{n_galaxies}_galaxy_test.pt"))
        test_image = test_galaxy["images"]

        with torch.no_grad():
            # get the estimated params
            trained_encoder.eval()
            (
                n_sources,
                locs,
                galaxy_params,
                log_fluxes,
                galaxy_bool,
            ) = trained_encoder.map_estimate(test_image.to(device))

        if not use_cuda:
            return

        # check number of galaxies is correct
        assert test_galaxy["n_galaxies"].item() == n_sources.item()

        # check locations are accurate.
        diff_locs = test_galaxy["locs"].sort(1)[0].to(device) - locs.sort(1)[0]
        diff_locs *= test_image.size(-1)
        assert diff_locs.abs().max() <= 2.5
