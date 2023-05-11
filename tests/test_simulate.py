from hydra.utils import instantiate
import torch
from bliss.predict import predict

class TestSimulate:
    def test_simulate(self, cfg):        
        sim_dataset = instantiate(cfg.simulator)
        
        for i in [8,11,19,27]:
            sim_tile = torch.load(cfg.paths.data + "/test_image/sim_tile" + str(i) + ".pt")
            sim_img = sim_dataset.simulate_image(sim_tile) 

            est_full = predict(cfg, sim_img[0], sim_img[1])
            est_tile = est_full.to_tile_params(cfg.encoder.tile_slen, 
                                               cfg.simulator.prior.max_sources)
            ttc = cfg.encoder.tiles_to_crop
            sim_galaxy_bools = sim_tile["galaxy_bools"][0,ttc:-ttc,ttc:-ttc,0,0]
            sim_star_bools = sim_tile["star_bools"][0,ttc:-ttc,ttc:-ttc,0,0]
                
            assert torch.equal(sim_galaxy_bools, est_tile["galaxy_bools"][0,:,:,0,0])
            assert torch.equal(sim_star_bools, est_tile["star_bools"][0,:,:,0,0])

            sim_star_fluxes = (sim_tile["star_fluxes"] * sim_tile["star_bools"])
            sim_galaxy_fluxes = (sim_tile["galaxy_params"] * sim_tile["galaxy_bools"])[:,:,:,:,0]
            sim_fluxes = (sim_star_fluxes[:,:,:,:,0] + sim_galaxy_fluxes)[0,ttc:-ttc,ttc:-ttc,0]

            est_star_fluxes = est_tile["star_fluxes"] * est_tile["star_bools"]
            est_galaxy_fluxes = (est_tile["galaxy_params"] * est_tile["galaxy_bools"])[:,:,:,:,0]
            est_fluxes = est_star_fluxes[0,:,:,0,0] + est_galaxy_fluxes[0,:,:,0]

            assert (est_fluxes-sim_fluxes).abs().sum()/(sim_fluxes.abs().sum()) < 0.1



        


        
       