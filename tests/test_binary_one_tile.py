import pytest
import torch


class TestGalaxyEncoder:
    @pytest.fixture(scope="class")
    def trained_encoder(self, decoder_setup, encoder_setup, device_setup):
        use_cuda = device_setup.use_cuda

        n_epochs = 120 if use_cuda else 1

        # draw galaxies + stars only in 10x10 center tile
        loc_min = 0.25
        loc_max = 0.75

        # encoder looks at 20x20 padded image, and detects galaxies/stars in 10x10 tile.
        ptile_slen = 20
        tile_slen = 10

        binary_dataset = decoder_setup.get_binary_dataset(
            slen=20,
            batch_size=128 if use_cuda else 2,
            n_batches=10 if use_cuda else 2,
            loc_min=loc_min,
            loc_max=loc_max,
            max_sources=2,
            mean_sources=1.5,
            min_sources=0,
        )
        trained_encoder = encoder_setup.get_trained_encoder(
            binary_dataset,
            n_epochs=n_epochs,
            max_detections=2,
            ptile_slen=ptile_slen,
            tile_slen=tile_slen,
            validation_plot_start=1000,  # do not create validation plots.
        )
        return trained_encoder.to(device_setup.device)

    def test_n_sources_and_locs(self, trained_encoder, paths, device_setup):
        use_cuda = device_setup.use_cuda
        device = device_setup.device

        test_batch = torch.load(paths["data"].joinpath(f"binary_test1.pt"))
        test_image = test_batch["images"]

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

        # check number of galaxies/stars is correct
        assert test_batch["n_sources"].item() == n_sources.item()
        assert test_batch["n_galaxies"].item() == galaxy_bool.sum()
        assert test_batch["n_stars"].item() == (1 - galaxy_bool).sum()

        # check detections are accurate.
        diff_locs = test_batch["locs"].sort(1)[0].to(device) - locs.sort(1)[0]
        diff_locs *= test_image.size(-1)
        assert diff_locs.abs().max() <= 0.5
