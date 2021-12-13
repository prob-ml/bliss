from einops.einops import rearrange
import torch
import pytorch_lightning as pl

from torch import Tensor
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import IterableDataset
from bliss.datasets.sdss import SloanDigitalSkySurvey
from bliss.models.binary import BinaryEncoder
from bliss.predict import Predict
from bliss.sleep import SleepPhase


class SdssBlendedGalaxies(pl.LightningDataModule, IterableDataset):
    image: Tensor

    def __init__(
        self,
        sleep_ckpt: str,
        binary_ckpt: str,
        sdss_dir="data/sdss",
        run=94,
        camcol=1,
        field=12,
        bands=(2,),
        bp=24,
        n_batches=10,
    ) -> None:
        super().__init__()
        sdss_data = SloanDigitalSkySurvey(
            sdss_dir=sdss_dir,
            run=run,
            camcol=camcol,
            fields=(field,),
            bands=bands,
            overwrite_cache=True,
            overwrite_fits_cache=True,
        )
        image = torch.from_numpy(sdss_data[0]["image"][0])
        self.image = rearrange(image, "h w -> 1 1 h w")
        self.bp = bp
        self.n_batches = n_batches

        sleep = SleepPhase.load_from_checkpoint(sleep_ckpt)
        image_encoder = sleep.image_encoder

        self.slen = 80

        binary_encoder = BinaryEncoder.load_from_checkpoint(binary_ckpt)
        self.predict_module = Predict(image_encoder.eval(), binary_encoder.eval())

    def __iter__(self):
        return self.batch_generator()

    def batch_generator(self):
        for _ in range(self.n_batches):
            yield self.get_batch()

    def get_batch(self):
        # Get chunk
        xlim, ylim = self.get_lims()
        chunk = self.image[:, :, ylim[0] : ylim[1], xlim[0] : xlim[1]]
        with torch.no_grad():
            tile_map, _ = self.predict_module.predict_on_image(chunk)
        batch = {
            "images": chunk,
            "n_sources": tile_map.n_sources,
            "locs": tile_map.locs,
            "galaxy_bool": tile_map.galaxy_bool,
            "star_bool": tile_map.star_bool,
            "fluxes": tile_map.fluxes,
            "log_fluxes": tile_map.log_fluxes,
            "slen": torch.tensor([self.slen]),
        }

        return batch

    def get_lims(self):
        x_start = torch.randint(low=0, high=(300 - self.slen), size=(1,))
        y_start = torch.randint(low=0, high=(300 - self.slen), size=(1,))
        xlim = (x_start + 1700 - self.bp, x_start + 1700 + self.slen + self.bp)
        ylim = (y_start + 200 - self.bp, y_start + 200 + self.slen + self.bp)

        return xlim, ylim

    def train_dataloader(self):
        return DataLoader(self, batch_size=None, num_workers=0)

    def val_dataloader(self):
        return DataLoader(self, batch_size=None, num_workers=0)

    def test_dataloader(self):
        return DataLoader(self, batch_size=None, num_workers=0)
