import numpy as np
import pytest
import torch
import pytorch_lightning as pl


class TestGalaxyEncoder:
    @pytest.fixture(scope="class")
    def trained_encoder(
        self,
        single_band_galaxy_decoder,
        fitted_psf,
        get_galaxy_dataset,
        get_trained_encoder,
    ):

        # draw galaxies only in 2x2 center tile
        loc_min = 0.4
        loc_max = 0.6

        # encoder looks at 10x10 padded tile, and detects galaxies in 2x2 tile.
        ptile_slen = 10
        edge_padding = 8

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
            n_epochs=100,
            max_detections=1,
            ptile_slen=ptile_slen,
            step=1,
            edge_padding=edge_padding,
        )
        return trained_encoder

    def test_n_sources_and_locs(self, trained_encoder):
        tr
