import pytest
import torch


class TestGalaxyEncoder:
    @pytest.fixture(scope="class")
    def trained_encoder(self, decoder_setup, encoder_setup, device_setup):
        use_cuda = device_setup.use_cuda

        n_epochs = 100 if use_cuda else 1
        # draw galaxies only in 2x2 center tile
        loc_min = 0.2
        loc_max = 0.8

        # encoder looks at 10x10 padded tile, and detects galaxies in 2x2 tile.
        ptile_slen = 8
        tile_slen = 2

        galaxy_dataset = decoder_setup.get_galaxy_dataset(
            slen=50,
            batch_size=32 if use_cuda else 2,
            n_images=320 if use_cuda else 2,
            loc_min=loc_min,
            loc_max=loc_max,
            max_sources=2,
            mean_sources=1,
            min_sources=1,
        )
        trained_encoder = encoder_setup.get_trained_encoder(
            galaxy_dataset,
            n_epochs=n_epochs,
            max_detections=2,
            ptile_slen=ptile_slen,
            tile_slen=tile_slen,
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
        assert diff_locs.abs().max() <= 0.5
