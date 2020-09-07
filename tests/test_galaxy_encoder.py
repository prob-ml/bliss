import pytest
import torch


class TestGalaxyEncoder:
    @pytest.fixture(scope="class")
    def trained_encoder(self, decoder_setup, encoder_setup, device_setup):
        use_cuda = device_setup.use_cuda

        n_epochs = 200 if use_cuda else 1

        # simulates either 1 or 2 galaxies in a 50 x 50 image
        # the input to the encoder is the 50 x 50 image
        slen = 50
        tile_slen = slen

        galaxy_dataset = decoder_setup.get_galaxy_dataset(
            slen=slen,
            tile_slen=tile_slen,
            batch_size=64 if use_cuda else 2,
            n_images=640 if use_cuda else 2,
            max_sources_per_tile=2,
            min_sources_per_tile=1,
            loc_max_per_tile = 0.8,
            loc_min_per_tile = 0.2,
            # this is so that prob(n_source = 1) \approx prob(n_source = 2) \approx = 0.5
            # under the poisson prior
            mean_sources_per_tile=1.67
        )
        trained_encoder = encoder_setup.get_trained_encoder(
            galaxy_dataset,
            n_epochs=n_epochs,
            max_detections=2,
            ptile_slen=tile_slen,
            tile_slen=tile_slen,
            validation_plot_start=0,
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

            # dim = 1 is the n_tiles dimension.
            # there is just one tile, so remove this dimension.
            locs = locs.squeeze(1)

        if not use_cuda:
            return

        # check number of galaxies is correct
        assert test_galaxy["n_galaxies"].item() == n_sources.item()

        # check locations are accurate.
        diff_locs = test_galaxy["locs"].sort(1)[0].to(device) - locs.sort(1)[0]
        diff_locs *= test_image.size(-1)
        assert diff_locs.abs().max() <= 2.5
