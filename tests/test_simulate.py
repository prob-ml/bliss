import numpy as np
import pytorch_lightning as pl
import torch
from hydra.utils import instantiate
from pytorch_lightning.utilities.types import EVAL_DATALOADERS
from torch.utils.data import DataLoader

from bliss.align import crop_image
from bliss.catalog import TileCatalog
from bliss.surveys.sdss import SloanDigitalSkySurvey as SDSS


class SDSSTest(pl.LightningDataModule):
    def __init__(self, image, background, cfg):
        super().__init__()
        self.items = [{"images": image, "background": background}]
        self.cfg = cfg

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        images, background = self.items[0].values()
        images = crop_image(images, self.cfg)
        background = crop_image(background, self.cfg)
        batch = {"images": images, "background": background}
        batch["images"] = batch["images"].squeeze(0)
        batch["background"] = batch["background"].squeeze(0)
        return DataLoader([batch], batch_size=1)


class TestSimulate:
    def test_simulate_and_predict(self, cfg, encoder):
        """Test simulating an image from a fixed catalog and making predictions on that catalog."""
        # load cached simulated catalog
        true_catalog = torch.load(cfg.paths.data + "/tests/test_image/dataset_0.pt")
        true_catalog = TileCatalog(4, true_catalog)

        # simulate image from catalog
        image_simulator = instantiate(cfg.simulator)
        # don't add noise to simulated image; with noise we intermittently generate what looks like
        # extra sources in the image, which causes the test to fail
        image_simulator.apply_noise = lambda img: img
        rcfs, rcf_indices = image_simulator.randomized_image_ids(true_catalog.n_sources.size(0))
        image, background, _, _ = image_simulator.simulate_image(true_catalog, rcfs, rcf_indices)

        # make predictions on simulated image
        true_catalog = true_catalog.to(cfg.predict.device)
        image = image.to(cfg.predict.device)
        background = background.to(cfg.predict.device)

        sdss_test = SDSSTest(image, background, cfg)
        encoder.eval()
        trainer = instantiate(cfg.predict.trainer)
        est_catalog = trainer.predict(encoder, datamodule=sdss_test)[0]["est_cat"].to(
            cfg.predict.device
        )

        # Compare predicted and true source types
        ttc = cfg.encoder.tiles_to_crop
        true_galaxy_bools = true_catalog.galaxy_bools[:, ttc:-ttc, ttc:-ttc]
        true_star_bools = true_catalog.star_bools[:, ttc:-ttc, ttc:-ttc]

        assert est_catalog.n_sources.sum() == 1
        assert est_catalog.star_bools.sum() == 1
        assert torch.equal(true_galaxy_bools, est_catalog.galaxy_bools)
        assert torch.equal(true_star_bools, est_catalog.star_bools)

        # Convert predicted fluxes from electron counts to nanomaggies for comparison
        flux_ratios = image_simulator.survey.flux_calibration_dict[(94, 1, 12)]
        est_catalog = nelec_to_nmgy_for_catalog(est_catalog, flux_ratios)

        # Compare predicted and true fluxes
        true_star_fluxes = true_catalog["star_fluxes"] * true_catalog.star_bools
        true_galaxy_fluxes = true_catalog["galaxy_fluxes"] * true_catalog.galaxy_bools
        true_fluxes = true_star_fluxes + true_galaxy_fluxes
        true_fluxes_crop = true_fluxes[0, ttc:-ttc, ttc:-ttc, 0, 2]

        est_star_fluxes = est_catalog["star_fluxes"] * est_catalog.star_bools
        est_galaxy_fluxes = est_catalog["galaxy_fluxes"] * est_catalog.galaxy_bools
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

        assert simulator.background.background.size()[0] == 2

    def test_multi_band(self, cfg, monkeypatch):
        """Test simulating data with multiple bands."""
        monkeypatch.delattr("requests.get")  # make sure we don't download anything
        simulator = instantiate(cfg.simulator)
        batch = simulator.get_batch()
        assert batch["images"].size(1) == len(SDSS.BANDS)
        assert batch["background"].size(1) == len(SDSS.BANDS)
