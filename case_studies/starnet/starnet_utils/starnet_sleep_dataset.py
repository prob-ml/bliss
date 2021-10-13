import torch
from bliss.datasets import simulated


class SimulatedStarnetDataset(simulated.SimulatedDataset):
    def __init__(
        self, decoder_kwargs, n_batches=10, batch_size=32, generate_device="cpu", testing_file=None
    ):
        super().__init__(decoder_kwargs, n_batches=10, batch_size=32, generate_device="cpu", testing_file=None)
        
        self.device = self.image_decoder.device
        
        # set the original background to be zero
        self.image_decoder.background_values = [0, 0]
        self.mean_background_vals = [686., 1123.]
    
    def sample_backgrounds(self, slen, batch_size): 
        mu = torch.Tensor(self.mean_background_vals).to(self.device).unsqueeze(0)
        sd = 50 
        
        normal_samples = torch.randn(size = (batch_size, 2)).to(self.device)
        
        sampled_background_vals = mu + normal_samples * sd
        
        background = background_shape = (batch_size, self.image_decoder.n_bands, slen, slen)
        background = torch.ones(*background_shape, device=self.image_decoder.device)
        
        background = background * sampled_background_vals.unsqueeze(-1).unsqueeze(-1)
        
        return background
        
    def get_batch(self):
        with torch.no_grad():
            batch = self.image_decoder.sample_prior(batch_size=self.batch_size)
            images, _ = self.image_decoder.render_images(
                batch["n_sources"],
                batch["locs"],
                batch["galaxy_bool"],
                batch["galaxy_params"],
                batch["fluxes"],
                add_noise=False,
            )
            
            
            # add in the random background
            background = self.sample_backgrounds(images.shape[-1], self.batch_size)
            images = images + background
            
            images = self.image_decoder._apply_noise(images)
            
            batch.update(
                {
                    "images": images,
                    "background": background,
                    "slen": torch.tensor([self.image_decoder.slen]),
                }
            )

        return batch
