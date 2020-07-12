import numpy as np
import pytest
import torch
import pytorch_lightning as pl


class TestGalaxyEncoder:
    @pytest.fixture(scope="class")
    def trained_encoder(
        self, fitted_psf, get_star_dataset, get_trained_star_encoder,
    ):
        star_dataset = get_star_dataset(fitted_psf, n_bands=1, slen=50, batch_size=32)
        trained_encoder = get_trained_star_encoder(star_dataset, n_epochs=100)
        return trained_encoder

    def test_n_sources(self,):
        pass
