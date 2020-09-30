import pytest
import torch


class TestBinaryEncoder:
    @pytest.fixture(scope="class")
    def trained_encoder(self, decoder_setup, encoder_setup, device_setup):
        use_cuda = device_setup.use_cuda

        n_epochs = 120 if use_cuda else 1

        # draw galaxies + stars only in 10x10 center tile
        # encoder looks at 20x20 padded image, and detects galaxies/stars in 10x10 tile.
        slen = 20

        binary_dataset = decoder_setup.get_binary_dataset(
            slen=slen,
            tile_slen=slen,
            ptile_padding=0,
            batch_size=128 if use_cuda else 2,
            n_batches=10 if use_cuda else 2,
            loc_min=0.25,
            loc_max=0.75,
            max_sources=2,
            mean_sources=1.5,
            min_sources=0,
            f_min=1e4,
            f_max=1e6,
            prob_galaxy=0.5,
        )
        trained_encoder = encoder_setup.get_trained_encoder(
            binary_dataset,
            n_epochs=n_epochs,
            max_detections=2,
            ptile_slen=slen,  # ensure 1 tile/ 1 padded tile only.
            tile_slen=slen,
            enc_hidden=256,
            enc_kern=3,
            enc_conv_c=20,
            validation_plot_start=0,
            background_pad_value=5000.0,
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

        # check predicted locations are accurate.
        diff_locs = test_batch["locs"].sort(1)[0].to(device) - locs.sort(1)[0]
        diff_locs *= test_image.size(-1)
        assert diff_locs.abs().max() <= 0.5
