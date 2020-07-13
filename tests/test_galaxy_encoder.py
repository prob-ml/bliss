import pytest
import torch

from bliss import use_cuda


class TestGalaxyEncoder:
    @pytest.fixture(scope="class")
    def trained_encoder(
        self, get_galaxy_dataset, get_trained_encoder,
    ):

        n_epochs = 100 if use_cuda else 1
        # draw galaxies only in 2x2 center tile
        loc_min = 0.4
        loc_max = 0.6

        # encoder looks at 10x10 padded tile, and detects galaxies in 2x2 tile.
        ptile_slen = 10
        edge_padding = 4

        galaxy_dataset = get_galaxy_dataset(
            slen=10,
            batch_size=32,
            n_images=128,
            loc_min=loc_min,
            loc_max=loc_max,
            max_sources=1,
            min_sources=1,
            mean_sources=1,
        )
        trained_encoder = get_trained_encoder(
            galaxy_dataset,
            n_epochs=n_epochs,
            max_detections=1,
            ptile_slen=ptile_slen,
            step=1,
            edge_padding=edge_padding,
        )
        return trained_encoder

    def test_n_sources_and_locs(self, data_path, device, trained_encoder):
        test_galaxy = torch.load(data_path.joinpath(f"test_small_galaxy.pt"))
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
            ) = trained_encoder.sample_encoder(
                test_image.to(device),
                n_samples=1,
                return_map_n_sources=True,
                return_map_source_params=True,
            )

        if not use_cuda:
            return

        diff_locs = test_galaxy["locs"].sort(1)[0].to(device) - locs.sort(1)[0]
        diff_locs *= test_image.size(-1)
        assert diff_locs.abs().max() <= 0.5
