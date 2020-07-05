import pytorch_lightning as pl
import pytest

from celeste import use_cuda
from celeste.datasets import galaxy_datasets
from celeste.models import galaxy_net


class TestGalaxyVAE:
    @pytest.fixture(scope="module")
    def trained_galaxy_vae(self, gpus, data_path):

        h5_file = data_path.joinpath("catsim_single_galaxies.hd5f")
        dataset = galaxy_datasets.H5Catalog(h5_file)
        n_epochs = 100 if use_cuda else 1
        trainer = pl.Trainer(
            gpus=gpus,
            min_epochs=n_epochs,
            max_epochs=n_epochs,
            limit_train_batches=20,
            logger=False,
            checkpoint_callback=False,
        )
        galaxy_vae = galaxy_net.OneCenteredGalaxy()

    def test_galaxy_vae(self):
        pass
