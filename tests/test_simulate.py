import numpy as np
import pytorch_lightning as pl
import torch
from hydra.utils import instantiate
from pytorch_lightning.utilities.types import EVAL_DATALOADERS
from torch.utils.data import DataLoader

from bliss.catalog import TileCatalog
from bliss.predict import crop_image, nelec_to_nmgy_for_catalog
from bliss.surveys.sdss import SloanDigitalSkySurvey as SDSS


class SDSSTest(pl.LightningDataModule):
    def __init__(self, image, background, cfg):
        super().__init__()
        self.items = [{"images": image, "background": background}]
        self.cfg = cfg

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        images, background = self.items[0].values()
        images, background = crop_image(images, background, self.cfg.predict.crop)
        batch = {"images": images, "background": background}
        batch["images"] = batch["images"].squeeze(0)
        batch["background"] = batch["background"].squeeze(0)
        return DataLoader([batch], batch_size=1)


class TestSimulate:
    def test_simulate(self, cfg, encoder):
        """Test simulating an image from a fixed catalog and making predictions on that catalog."""
        # load cached simulated catalog
        sim_dataset = instantiate(cfg.simulator)
        sim_tile = torch.load(cfg.paths.data + "/tests/test_image/dataset_0.pt")
        sim_tile = TileCatalog(4, sim_tile)

        # simulate image from catalog
        rcfs, rcf_indices = sim_dataset.randomized_image_ids(sim_tile.n_sources.size(0))
        image, background, _, _ = sim_dataset.simulate_image(sim_tile, rcfs, rcf_indices)

        # make predictions on simulated image
        sim_tile = sim_tile.to(cfg.predict.device)
        image = image.to(cfg.predict.device)
        background = background.to(cfg.predict.device)

        sdss_test = SDSSTest(image, background, cfg)
        encoder.eval()
        trainer = instantiate(cfg.predict.trainer)
        est_tile = trainer.predict(encoder, datamodule=sdss_test)[0]["est_cat"].to(
            cfg.predict.device
        )

        # Compare predicted and true source types
        ttc = cfg.encoder.tiles_to_crop
        sim_galaxy_bools = sim_tile.galaxy_bools[:, ttc:-ttc, ttc:-ttc]
        sim_star_bools = sim_tile.star_bools[:, ttc:-ttc, ttc:-ttc]

        assert torch.equal(sim_galaxy_bools, est_tile.galaxy_bools)
        assert torch.equal(sim_star_bools, est_tile.star_bools)

        # Convert predicted fluxes from electron counts to nanomaggies for comparison
        flux_ratios = sim_dataset.survey.flux_calibration_dict[(94, 1, 12)]
        est_tile = nelec_to_nmgy_for_catalog(est_tile, flux_ratios)

        # Compare predicted and true fluxes
        sim_star_fluxes = sim_tile["star_fluxes"] * sim_tile.star_bools
        sim_galaxy_fluxes = sim_tile["galaxy_fluxes"] * sim_tile.galaxy_bools
        sim_fluxes = sim_star_fluxes + sim_galaxy_fluxes
        sim_fluxes_crop = sim_fluxes[0, ttc:-ttc, ttc:-ttc, 0, 2]

        est_star_fluxes = est_tile["star_fluxes"] * est_tile.star_bools
        est_galaxy_fluxes = est_tile["galaxy_fluxes"] * est_tile.galaxy_bools
        est_fluxes = est_star_fluxes + est_galaxy_fluxes
        est_fluxes = est_fluxes[0, :, :, 0, 2]

        assert (est_fluxes - sim_fluxes_crop).abs().sum() / (sim_fluxes_crop.abs().sum()) < 1.0

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
