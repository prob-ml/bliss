import pytorch_lightning as pl
import torch
from torch.utils.data.dataloader import DataLoader
from torch.nn import functional as F
from einops.einops import rearrange
from tqdm import tqdm

from bliss.datasets.sdss import SloanDigitalSkySurvey
from bliss.models.binary import BinaryEncoder
from bliss.encoder import Encoder
from bliss.sleep import SleepPhase


class SdssBlendedGalaxies(pl.LightningDataModule):
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
        h_start=200,
        w_start=1000,
        scene_size=1000,
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
        image = rearrange(image, "h w -> 1 1 h w")
        self.bp = bp
        self.slen = 80 + 2 * bp
        self.n_batches = n_batches
        image = image[
            :,
            :,
            (h_start - self.bp) : (h_start + scene_size + self.bp),
            (w_start - self.bp) : (w_start + scene_size + self.bp),
        ]

        sleep = SleepPhase.load_from_checkpoint(sleep_ckpt)
        image_encoder = sleep.image_encoder

        binary_encoder = BinaryEncoder.load_from_checkpoint(binary_ckpt)
        self.encoder = Encoder(image_encoder.eval(), binary_encoder.eval())
        self.chunks, self.catalogs = self.prerender_chunks(image)

    def __len__(self):
        return len(self.catalogs)

    def __getitem__(self, idx):
        chunk = self.chunks[idx]
        tile_map = self.catalogs[idx]
        return {
            "images": chunk,
            **tile_map,
            "slen": torch.tensor([self.slen]),
        }

    def prerender_chunks(self, image):
        kernel_size = self.slen + 2 * self.bp
        chunks = F.unfold(image, kernel_size=kernel_size, stride=self.slen)
        chunks = rearrange(
            chunks,
            "b (c h w) n -> (b n) c h w",
            c=image.shape[1],
            h=kernel_size,
            w=kernel_size,
        )
        catalogs = []
        chunks_with_galaxies = []
        with torch.no_grad():
            for chunk in tqdm(chunks):
                image_ptiles = self.encoder.get_images_in_ptiles(chunk.unsqueeze(0))
                tile_map = self.encoder.max_a_post(image_ptiles)
                if tile_map["galaxy_bool"].sum() > 0:
                    catalogs.append(tile_map)
                    chunks_with_galaxies.append(chunk)
        chunks_with_galaxies = torch.stack(chunks_with_galaxies, dim=0)
        print(f"n_galaxies: {chunks_with_galaxies.shape[0]}")
        return chunks_with_galaxies, catalogs

    def train_dataloader(self):
        return DataLoader(self, batch_size=1, num_workers=0)

    def val_dataloader(self):
        return DataLoader(self, batch_size=1, num_workers=0)

    def test_dataloader(self):
        return DataLoader(self, batch_size=1, num_workers=0)
