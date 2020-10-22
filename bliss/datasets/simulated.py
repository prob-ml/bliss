from omegaconf import DictConfig
import torch
from torch.utils.data import IterableDataset

from bliss.models.decoder import ImageDecoder


class SimulatedDataset(IterableDataset):
    def __init__(self, cfg: DictConfig):
        super(SimulatedDataset, self).__init__()

        self.n_batches = cfg.dataset.params.n_batches
        self.batch_size = cfg.dataset.params.batch_size
        self.image_decoder = ImageDecoder(**cfg.model.decoder.params)

    def __iter__(self):
        return self.batch_generator()

    def batch_generator(self):
        for i in range(self.n_batches):
            yield self.get_batch()

    def get_batch(self):
        with torch.no_grad():
            params = self.image_decoder.sample_prior(batch_size=self.batch_size)
            images = self.image_decoder.render_images(
                params["n_sources"],
                params["locs"],
                params["galaxy_bool"],
                params["galaxy_params"],
                params["fluxes"],
            )
            params.update(
                {"images": images, "background": self.image_decoder.background}
            )

        return params
