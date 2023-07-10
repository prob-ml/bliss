import numpy as np
import pytorch_lightning as pl
import torch
from hydra.utils import instantiate
from pytorch_lightning.utilities.types import EVAL_DATALOADERS
from torch.utils.data import DataLoader

from bliss.catalog import TileCatalog
from bliss.predict import crop_image
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
        # loads single r-band model with correct number of outputs
        sim_dataset = instantiate(cfg.simulator)
        sim_tile = torch.load(cfg.paths.data + "/tests/test_image/dataset_0.pt")
        # NOTE: sim_tile is a batch of 4 images
        sim_tile = TileCatalog(4, sim_tile)

        _, rcf_indices = sim_dataset.randomized_image_ids(
            sim_tile.n_sources.size(0)
        )  # noqa: WPS437
        image, background, _, _ = sim_dataset.simulate_image(sim_tile, rcf_indices)

        # move data to the device the encoder is on
        sim_tile = sim_tile.to(cfg.predict.device)
        image = image.to(cfg.predict.device)
        background = background.to(cfg.predict.device)

        sdss_test = SDSSTest(image, background, cfg)
        encoder.eval()
        trainer = instantiate(cfg.predict.trainer)
        est_tile = trainer.predict(encoder, datamodule=sdss_test)[0]["est_cat"].to(
            cfg.predict.device
        )
        ttc = cfg.encoder.tiles_to_crop
        sim_galaxy_bools = sim_tile.galaxy_bools[:, ttc:-ttc, ttc:-ttc]
        sim_star_bools = sim_tile.star_bools[:, ttc:-ttc, ttc:-ttc]

        assert torch.equal(sim_galaxy_bools, est_tile.galaxy_bools)
        assert torch.equal(sim_star_bools, est_tile.star_bools)

        sim_star_fluxes = sim_tile["star_fluxes"] * sim_tile.star_bools
        sim_galaxy_fluxes = (sim_tile["galaxy_fluxes"] * sim_tile.galaxy_bools)[..., 0]
        sim_fluxes = sim_star_fluxes[:, :, :, :, 0] + sim_galaxy_fluxes
        sim_fluxes_crop = sim_fluxes[0, ttc:-ttc, ttc:-ttc, 0]

        est_star_fluxes = est_tile["star_fluxes"] * est_tile.star_bools
        est_galaxy_fluxes = (est_tile["galaxy_fluxes"] * est_tile.galaxy_bools)[..., 0]
        est_fluxes = est_star_fluxes[0, :, :, 0, 0] + est_galaxy_fluxes[0, :, :, 0]

        # NOTE: Considers ALL 5 predicted fluxes and ALL four images in batch
        assert (est_fluxes - sim_fluxes_crop).abs().sum() / (sim_fluxes_crop.abs().sum()) < 0.3

    def test_multi_background(self, cfg, monkeypatch):
        """Test loading backgrounds and PSFs from multiple fields works."""
        monkeypatch.delattr("requests.get")  # make sure we don't download anything
        sdss_fields = [
            {"run": 94, "camcol": 1, "fields": [12]},
            {"run": 3900, "camcol": 6, "fields": [269]},
        ]

        simulator = instantiate(cfg.simulator, survey={"sdss_fields": sdss_fields})
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
