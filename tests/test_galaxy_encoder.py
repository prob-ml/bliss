import pytest
import torch

from bliss import use_cuda


class TestGalaxyEncoder:
    @pytest.fixture(scope="class")
    def trained_encoder(
        self, get_galaxy_dataset, get_trained_encoder,
    ):

        n_epochs = 150 if use_cuda else 1
        # draw galaxies only in 2x2 center tile
        loc_min = 0.2
        loc_max = 0.8

        # encoder looks at 10x10 padded tile, and detects galaxies in 2x2 tile.
        ptile_slen = 8
        edge_padding = 3

        galaxy_dataset = get_galaxy_dataset(
            slen=50,
            batch_size=32,
            n_images=320,
            loc_min=loc_min,
            loc_max=loc_max,
            max_sources=2,
            mean_sources=1.5,
            min_sources=1,
        )
        trained_encoder = get_trained_encoder(
            galaxy_dataset,
            n_epochs=n_epochs,
            max_detections=2,
            ptile_slen=ptile_slen,
            step=2,
            edge_padding=edge_padding,
        )
        return trained_encoder

    @pytest.mark.parametrize("n_galaxies", ["1", "3"])
    def test_n_sources_and_locs(self, n_galaxies, data_path, device, trained_encoder):
        test_galaxy = torch.load(data_path.joinpath(f"{n_galaxies}_galaxy_test.pt"))
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

        # check number of galaxies is correct
        assert test_galaxy["n_galaxies"].item() == galaxy_bool.sum().item()

        # check locations are accurate.
        diff_locs = test_galaxy["locs"].sort(1)[0].to(device) - locs.sort(1)[0]
        diff_locs *= test_image.size(-1)
        assert diff_locs.abs().max() <= 0.5
