from pathlib import Path

import pytorch_lightning as pl
import torch
from hydra.utils import instantiate
from torch.utils.data import DataLoader

from bliss.catalog import TileCatalog


class MockSDSS(pl.LightningDataModule):
    def __init__(self, image, psf_params):
        super().__init__()
        self.one_batch = {
            "images": image.squeeze(0),
            "psf_params": psf_params.squeeze(0),
        }

    def predict_dataloader(self):
        return DataLoader([self.one_batch], batch_size=1)


class TestSimulate:
    def test_simulate_and_predict(self, cfg):
        """Test simulating an image from a fixed catalog and making predictions on that catalog."""
        # load cached simulated catalog
        true_catalog = torch.load(
            f"{cfg.paths.test_data}/test_image/dataset_0.pt", weights_only=False
        )
        true_catalog["fluxes"][0, 10, 10] = 10.0
        true_catalog = TileCatalog(true_catalog)

        # simulate image from catalog
        decoder = instantiate(cfg.decoder)
        # don't add noise to simulated image; with noise we intermittently generate what looks like
        # extra sources in the image, which causes the test to fail
        image, psf_params = decoder.render_images(true_catalog)

        # make predictions on simulated image
        true_catalog = true_catalog.to(cfg.predict.device)
        image = image.to(cfg.predict.device)
        psf_params = psf_params.to(cfg.predict.device)

        sdss = MockSDSS(image, psf_params)

        encoder = instantiate(cfg.encoder).to(cfg.predict.device)
        enc_state_dict = torch.load(
            cfg.predict.weight_save_path, map_location=cfg.predict.device, weights_only=False
        )
        encoder.load_state_dict(enc_state_dict)
        encoder.eval()

        trainer = instantiate(cfg.predict.trainer)
        mode_cat = trainer.predict(encoder, datamodule=sdss)[0]
        mode_cat = mode_cat.to(cfg.predict.device)

        # Compare predicted and true source types
        assert mode_cat["n_sources"].sum() == 1
        assert mode_cat.star_bools.sum() == 1
        assert torch.equal(true_catalog.galaxy_bools, mode_cat.galaxy_bools)
        assert torch.equal(true_catalog.star_bools, mode_cat.star_bools)

        # Compare predicted and true fluxes
        true_fluxes = true_catalog.on_fluxes[0, :, :, 0, 2]
        est_fluxes = mode_cat.on_fluxes[0, :, :, 0, 2]

        assert (est_fluxes - true_fluxes).abs().sum() / (true_fluxes.abs().sum()) < 1.0

    def test_render_images(self, cfg):
        with open(Path(cfg.paths.test_data) / "sdss_preds.pt", "rb") as f:
            test_cat = torch.load(f, weights_only=False)

        true_tile_cat = TileCatalog(test_cat).to("cpu")

        # first we'll render the image from the catalog
        # these are sky subtracted images in physical units (nanomaggies)
        decoder = instantiate(cfg.decoder)
        rendered_image, _psf_params = decoder.render_images(true_tile_cat)

        # then we'll compare the reconstructed image to the true fluxes
        source_fluxes = true_tile_cat.on_fluxes.sum([0, 1, 2, 3])
        rendered_fluxes = rendered_image.sum([0, 2, 3])
        # some flux is outside of the image bounds
        assert (rendered_fluxes <= source_fluxes).all()
        # we should capture the majority of the flux, at least in the middle bands
        assert (rendered_fluxes[1:4] >= 0.5 * source_fluxes[1:4]).all()
