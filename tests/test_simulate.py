import torch
from hydra.utils import instantiate

from bliss.predict import predict


class TestSimulate:
    def test_simulate(self, cfg):
        sim_dataset = instantiate(cfg.simulator)
        tile_slen = cfg.encoder.tile_slen
        max_sources = cfg.simulator.prior.max_sources

        for i in range(4):
            sim_tile = torch.load(cfg.paths.data + "/test_image/sim_tile" + str(i) + ".pt")
            image, background = sim_dataset.simulate_image(sim_tile)

            # move data to the device the encoder is on
            sim_tile = sim_tile.to(cfg.predict.device)
            image = image.to(cfg.predict.device)
            background = background.to(cfg.predict.device)

            est_full = predict(cfg, image, background)
            est_tile = est_full.to_tile_params(tile_slen, max_sources)
            ttc = cfg.encoder.tiles_to_crop
            sim_galaxy_bools = sim_tile["galaxy_bools"][:, ttc:-ttc, ttc:-ttc]
            sim_star_bools = sim_tile["star_bools"][:, ttc:-ttc, ttc:-ttc]

            assert torch.equal(sim_galaxy_bools, est_tile["galaxy_bools"])
            assert torch.equal(sim_star_bools, est_tile["star_bools"])

            sim_star_fluxes = sim_tile["star_fluxes"] * sim_tile["star_bools"]
            sim_galaxy_params = sim_tile["galaxy_params"] * sim_tile["galaxy_bools"]
            sim_galaxy_fluxes = sim_galaxy_params[:, :, :, :, 0]
            sim_fluxes = sim_star_fluxes[:, :, :, :, 0] + sim_galaxy_fluxes
            sim_fluxes_crop = sim_fluxes[0, ttc:-ttc, ttc:-ttc, 0]

            est_star_fluxes = est_tile["star_fluxes"] * est_tile["star_bools"]
            est_galaxy_params = est_tile["galaxy_params"] * est_tile["galaxy_bools"]
            est_galaxy_fluxes = est_galaxy_params[:, :, :, :, 0]
            est_fluxes = est_star_fluxes[0, :, :, 0, 0] + est_galaxy_fluxes[0, :, :, 0]

            assert (est_fluxes - sim_fluxes_crop).abs().sum() / (sim_fluxes_crop.abs().sum()) < 0.1
