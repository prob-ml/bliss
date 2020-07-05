import pytorch_lightning as pl
import pytest
import torch
import numpy as np

from celeste import use_cuda
from celeste.datasets import galaxy_datasets
from celeste.models import galaxy_net


class TestGalaxyVAE:
    @pytest.fixture(scope="module")
    def trained_galaxy_vae(self, gpus, data_path):

        h5_file = data_path.joinpath("catsim_single_galaxies.hdf5")
        dataset = galaxy_datasets.H5Catalog(h5_file, slen=51, n_bands=1)
        n_epochs = 100 if use_cuda else 1
        trainer = pl.Trainer(
            gpus=gpus,
            min_epochs=n_epochs,
            max_epochs=n_epochs,
            limit_train_batches=20,
            profiler=None,
            logger=False,
            checkpoint_callback=False,
            limit_val_batches=0.0,  # no validation.
        )

        # TODO: Allow tt_split ==0 if no validation.
        galaxy_vae = galaxy_net.OneCenteredGalaxy(
            dataset,
            slen=51,
            n_bands=1,
            latent_dim=8,
            num_workers=0,
            batch_size=64,
            tt_split=0.1,
        )

        trainer.fit(galaxy_vae)
        galaxy_vae.freeze()
        galaxy_vae.eval()
        return galaxy_vae

    def test_galaxy_vae(self, data_path, trained_galaxy_vae, device):
        galaxy_image = torch.load(data_path.joinpath("1_catsim_galaxy.pt"))
        background = torch.from_numpy(
            np.load(data_path.joinpath("background_galaxy_single_band_i.npy"))
        )

        galaxy_image = galaxy_image.to(device)
        galaxy_image = galaxy_image.reshape(1, 1, galaxy_image.size(-1), -1)

        with torch.no_grad():
            pred_image, _, _ = trained_galaxy_vae(galaxy_image, background)

        residual = (galaxy_image - pred_image) / torch.sqrt(galaxy_image)

        # only expect tests to pass in cuda:
        if not use_cuda:
            return

        # check residuals follow gaussian noise, most pixels are between 68%
        n_pixels = residual.size(-1) ** 2
        assert (residual.abs() <= 1).sum() >= n_pixels * 0.5
