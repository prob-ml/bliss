from hydra.utils import instantiate
import torch
from bliss.predict import prepare_image


# If we simulate small images with various combinations of one, two, 
# or three stars and galaxies, the encoder with pretrained 
# weights (currently stored in data/pretrained_models) correctly predicts 
# the locations and properties of these sources.

class TestSimulate:
    def test_simulate(self, cfg):
        img_prior = instantiate(cfg.simulator.prior)
        sim_dataset = instantiate(cfg.simulator)
        sim_tile = img_prior.sample_prior()

        encoder = instantiate(cfg.encoder).to(cfg.predict.device)
        enc_state_dict = torch.load(cfg.predict.weight_save_path)
        encoder.load_state_dict(enc_state_dict)
        encoder.eval()

        sim_img = sim_dataset.simulate_image(sim_tile)
        batch = {
        "images": sim_img[0],
        "background": sim_img[1],
        }
        # check light source and flux

        with torch.no_grad():
            pred = encoder.encode_batch(batch)
            est_cat = encoder.variational_mode(pred)

        


        
       