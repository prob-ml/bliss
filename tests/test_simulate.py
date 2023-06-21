import numpy as np
import pytorch_lightning as pl
import torch
from hydra.utils import instantiate
from omegaconf.dictconfig import DictConfig
from pytorch_lightning.utilities.types import EVAL_DATALOADERS
from torch.utils.data import DataLoader

from bliss.simulator.background import SimulatedSDSSBackground
from bliss.surveys.sdss import prepare_batch


class SDSSTest(pl.LightningDataModule):
    def __init__(self, image, background, cfg):
        super().__init__()
        self.items = [{"images": image, "background": background}]
        self.cfg = cfg

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        images, background = self.items[0].values()
        batch = prepare_batch(images, background, self.cfg.predict.dataset.predict_crop)
        return DataLoader([batch], batch_size=1)


class TestSimulate:
    def test_simulate(self, cfg):
        # temporary addition until new tile catalogs are generated
        # loads single r-band model with correct number of outputs
        cfg.simulator.sdss_fields.bands = [2]
        cfg.encoder.bands = [2]
        cfg.encoder.z_score = False
        sim_dataset = instantiate(cfg.simulator)
        tile_slen = cfg.encoder.tile_slen
        max_sources = cfg.simulator.prior.max_sources

        for i in range(4):
            sim_tile = torch.load(cfg.paths.data + "/test_image/sim_tile" + str(i) + ".pt")
            _, rcf_indices = sim_dataset.get_random_rcf(sim_tile.n_sources.size(0))  # noqa: WPS437
            sim_tile["source_type"] = sim_tile.pop("galaxy_bools")
            sim_tile.pop("star_bools")
            image, background, _, _ = sim_dataset.simulate_image(sim_tile, rcf_indices)

            # move data to the device the encoder is on
            sim_tile = sim_tile.to(cfg.predict.device)
            image = image.to(cfg.predict.device)
            background = background.to(cfg.predict.device)

            sdss_test = SDSSTest(image, background, cfg)
            encoder = instantiate(cfg.encoder).to(cfg.predict.device)
            enc_state_dict = torch.load(cfg.predict.weight_save_path)
            encoder.load_state_dict(enc_state_dict)
            encoder.eval()
            trainer = instantiate(cfg.predict.trainer)
            est_full = trainer.predict(encoder, datamodule=sdss_test)[0]["est_cat"]
            est_tile = est_full.to_tile_params(tile_slen, max_sources)
            ttc = cfg.encoder.tiles_to_crop
            sim_galaxy_bools = sim_tile.galaxy_bools[:, ttc:-ttc, ttc:-ttc]
            sim_star_bools = sim_tile.star_bools[:, ttc:-ttc, ttc:-ttc]

            assert torch.equal(sim_galaxy_bools, est_tile.galaxy_bools)
            assert torch.equal(sim_star_bools, est_tile.star_bools)

            sim_star_fluxes = sim_tile["star_fluxes"] * sim_tile.star_bools
            sim_galaxy_params = sim_tile["galaxy_params"] * sim_tile.galaxy_bools
            sim_galaxy_fluxes = sim_galaxy_params[:, :, :, :, 0]
            sim_fluxes = sim_star_fluxes[:, :, :, :, 0] + sim_galaxy_fluxes
            sim_fluxes_crop = sim_fluxes[0, ttc:-ttc, ttc:-ttc, 0]

            est_star_fluxes = est_tile["star_fluxes"] * est_tile.star_bools
            est_galaxy_params = est_tile["galaxy_params"] * est_tile.galaxy_bools
            est_galaxy_fluxes = est_galaxy_params[:, :, :, :, 0]
            est_fluxes = est_star_fluxes[0, :, :, 0, 0] + est_galaxy_fluxes[0, :, :, 0]

            assert (est_fluxes - sim_fluxes_crop).abs().sum() / (sim_fluxes_crop.abs().sum()) < 0.1

    def test_multi_background(self, cfg):
        """Test loading backgrounds and PSFs from multiple fields works."""
        sdss_fields = {
            "dir": cfg.simulator.sdss_fields.dir,
            "bands": cfg.simulator.sdss_fields.bands,
            "field_list": [
                {"run": 94, "camcol": 1, "fields": [12]},
                {"run": 3900, "camcol": 6, "fields": [269]},
            ],
        }
        decoder = {"sdss_fields": sdss_fields}  # override isn't passed recursively

        simulator = instantiate(cfg.simulator, sdss_fields=sdss_fields, decoder=decoder)
        assert np.all(simulator.rcf_list == np.array([[94, 1, 12], [3900, 6, 269]]))
        assert (94, 1, 12) in simulator.image_decoder.psf_galsim.keys()
        assert (3900, 6, 269) in simulator.image_decoder.psf_galsim.keys()

        # manually construct background since testing_config uses ConstantBackground
        bg_cfg = DictConfig(sdss_fields)
        background = SimulatedSDSSBackground(bg_cfg)
        assert background.background.size()[0] == 2

    def test_multi_band(self, cfg):
        """Test simulating data with multiple bands."""
        cfg.simulator.sdss_fields.bands = [2, 3, 4]  # override default with multiple bands
        simulator = instantiate(cfg.simulator)
        batch = simulator.get_batch()
        assert batch["images"].size(1) == 3
        assert batch["background"].size(1) == 3
