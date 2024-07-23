from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
from hydra.utils import instantiate
from pytorch_lightning.utilities.types import EVAL_DATALOADERS
from torch.utils.data import DataLoader

from bliss.catalog import TileCatalog
from bliss.surveys.sdss import SloanDigitalSkySurvey as SDSS
from bliss.surveys.sdss import nelec_to_nmgy_for_catalog


class MockSDSS(pl.LightningDataModule):
    def __init__(self, image, psf_params):
        super().__init__()
        self.one_batch = {
            "images": image.squeeze(0),
            "psf_params": psf_params.squeeze(0),
        }

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader([self.one_batch], batch_size=1)


class TestSimulate:
    def test_simulate_and_predict(self, cfg, encoder):
        """Test simulating an image from a fixed catalog and making predictions on that catalog."""
        # load cached simulated catalog
        true_catalog = torch.load(cfg.paths.test_data + "/test_image/dataset_0.pt")
        true_catalog["star_fluxes"][0, 10, 10] = 10.0
        true_catalog = TileCatalog(4, true_catalog)

        # simulate image from catalog
        image_simulator = instantiate(cfg.simulator)
        # don't add noise to simulated image; with noise we intermittently generate what looks like
        # extra sources in the image, which causes the test to fail
        rcfs, rcf_indices = image_simulator.randomized_image_ids(true_catalog["n_sources"].size(0))
        image, psf_params = image_simulator.image_decoder.render_images(
            true_catalog, rcfs, rcf_indices, add_dither=False, add_noise=False
        )

        # make predictions on simulated image
        true_catalog = true_catalog.to(cfg.predict.device)
        image = image.to(cfg.predict.device)
        psf_params = psf_params.to(cfg.predict.device)

        sdss = MockSDSS(image, psf_params)
        encoder.eval()
        trainer = instantiate(cfg.predict.trainer)
        mode_cat = trainer.predict(encoder, datamodule=sdss)[0]["mode_cat"]
        mode_cat = mode_cat.to(cfg.predict.device)

        # Compare predicted and true source types
        assert mode_cat["n_sources"].sum() == 1
        assert mode_cat.star_bools.sum() == 1
        assert torch.equal(true_catalog.galaxy_bools, mode_cat.galaxy_bools)
        assert torch.equal(true_catalog.star_bools, mode_cat.star_bools)

        # Convert predicted fluxes from electron counts to nanomaggies for comparison
        flux_ratios = image_simulator.survey.flux_calibration_dict[(94, 1, 12)]
        mode_cat = nelec_to_nmgy_for_catalog(mode_cat, flux_ratios)

        # Compare predicted and true fluxes
        true_star_fluxes = true_catalog["star_fluxes"] * true_catalog.star_bools
        true_galaxy_fluxes = true_catalog["galaxy_fluxes"] * true_catalog.galaxy_bools
        true_fluxes = true_star_fluxes + true_galaxy_fluxes
        true_fluxes_crop = true_fluxes[0, :, :, 0, 2]

        est_star_fluxes = mode_cat["star_fluxes"] * mode_cat.star_bools
        est_galaxy_fluxes = mode_cat["galaxy_fluxes"] * mode_cat.galaxy_bools
        est_fluxes = est_star_fluxes + est_galaxy_fluxes
        est_fluxes = est_fluxes[0, :, :, 0, 2]

        assert (est_fluxes - true_fluxes_crop).abs().sum() / (true_fluxes_crop.abs().sum()) < 1.0

    def test_multi_background(self, cfg, monkeypatch):
        """Test loading backgrounds and PSFs from multiple fields works."""
        monkeypatch.delattr("requests.get")  # make sure we don't download anything
        sdss_fields = [
            {"run": 94, "camcol": 1, "fields": [12]},
            {"run": 3900, "camcol": 6, "fields": [269]},
        ]

        simulator = instantiate(cfg.simulator, survey={"fields": sdss_fields})
        assert np.all(simulator.image_ids == np.array([[94, 1, 12], [3900, 6, 269]]))
        assert (94, 1, 12) in simulator.image_decoder.psf_galsim.keys()
        assert (3900, 6, 269) in simulator.image_decoder.psf_galsim.keys()

    def test_multi_band(self, cfg, monkeypatch):
        """Test simulating data with multiple bands."""
        monkeypatch.delattr("requests.get")  # make sure we don't download anything
        simulator = instantiate(cfg.simulator)
        batch = simulator.get_batch()
        assert batch["images"].size(1) == len(SDSS.BANDS)

    def test_render_images(self, cfg, decoder):
        with open(Path(cfg.paths.test_data) / "sdss_preds.pt", "rb") as f:
            test_datum = torch.load(f)

        true_tile_cat = TileCatalog(cfg.simulator.prior.tile_slen, test_datum["catalog"]).to("cpu")

        # first we'll render the image from the catalog
        rcfs = [(94, 1, 12)]
        rcfs_indices = torch.zeros(1, dtype=torch.long)
        # these are sky subtracted images in physical units (nanomaggies)
        rendered_image, _psf_params = decoder.render_images(
            true_tile_cat, rcfs, rcfs_indices, add_noise=False
        )

        # then we'll compare the reconstructed image to the true fluxes
        source_fluxes = true_tile_cat.on_nmgy.sum([0, 1, 2, 3])
        rendered_fluxes = rendered_image.sum([0, 2, 3])
        # some flux is outside of the image bounds
        assert (rendered_fluxes <= source_fluxes).all()
        # we should capture the majority of the flux, at least in the middle bands
        assert (rendered_fluxes[1:4] >= 0.5 * source_fluxes[1:4]).all()
