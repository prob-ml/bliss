from pathlib import Path
from typing import Dict, Optional, Tuple

import pytorch_lightning as pl
import torch
from torch.utils.data.dataloader import DataLoader
from torch.nn import functional as F
from torch import Tensor
from einops.einops import rearrange
from tqdm import tqdm

from bliss.datasets.sdss import SloanDigitalSkySurvey
from bliss.models.binary import BinaryEncoder
from bliss.encoder import Encoder
from bliss.sleep import SleepPhase


class SdssBlendedGalaxies(pl.LightningDataModule):
    """Find and catalog chunks of SDSS images with galaxies in them.

    This dataset takes an SDSS image, splits it into chunks, finds the locations,
    classifies each object as a star or galaxy, and only keeps chunks with at least one galaxy.
    The intended purpose is for learning a distribution of galaxies from real images.

    """

    def __init__(
        self,
        sleep: SleepPhase,
        binary_encoder: BinaryEncoder,
        sleep_ckpt: str,
        binary_ckpt: str,
        sdss_dir: str = "data/sdss",
        run: int = 94,
        camcol: int = 1,
        field: int = 12,
        bands: Tuple[int, ...] = (2,),
        bp: int = 24,
        slen: int = 80,
        h_start: Optional[int] = None,
        w_start: Optional[int] = None,
        scene_size: Optional[int] = None,
        stride_factor: float = 0.5,
        cache_path: Optional[str] = None,
        prerender_device: str = "cpu",
    ) -> None:
        """Initializes SDSSBlendedGalaxies.

        Args:
            sleep: A SleepPhase model for getting the locations of sources.
            binary_encoder: A BinaryEncoder model.
            sleep_ckpt: Path of saved state_dict for sleep-phase-trained encoder.
            binary_ckpt: Path of saved state_dict for binary encoder.
            sdss_dir: Location of data storage for SDSS. Defaults to "data/sdss".
            run: SDSS run.
            camcol: SDSS camcol.
            field: SDSS field.
            bands: SDSS bands of image to use.
            bp: How much border padding around each chunk.
            slen: Side-length of each chunk.
            h_start: Starting height-point of image. If None, start at `bp`.
            w_start: Starting width-point of image. If None, start at `bp`.
            scene_size: Total size of the scene to use. If None, use maximum possible size.
            stride_factor: How much should chunks overlap? If 1.0, no overlap.
            cache_path: If not None, path where cached chunks and catalogs are saved.
            prerender_device: Device to use to prerender chunks.
        """
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
        self.slen = slen + 2 * bp
        self.kernel_size = self.slen + 2 * self.bp
        self.stride = int(self.slen * stride_factor)
        assert self.stride > 0
        self.prerender_device = prerender_device

        if h_start is None:
            h_start = self.bp
        if w_start is None:
            w_start = self.bp
        assert h_start >= self.bp
        assert w_start >= self.bp

        if scene_size is None:
            scene_size = min(image.shape[2] - h_start, image.shape[3] - w_start) - self.bp
        image = image[
            :,
            :,
            (h_start - self.bp) : (h_start + scene_size + self.bp),
            (w_start - self.bp) : (w_start + scene_size + self.bp),
        ]

        sleep.load_state_dict(torch.load(sleep_ckpt, map_location=torch.device("cpu")))
        image_encoder = sleep.image_encoder

        binary_encoder.load_state_dict(torch.load(binary_ckpt, map_location=torch.device("cpu")))
        self.encoder = Encoder(image_encoder.eval(), binary_encoder.eval())
        cache_file = (
            Path(cache_path + f"_h{h_start}w{w_start}s{scene_size}.pt")
            if cache_path is not None
            else None
        )
        if (cache_file is not None) and cache_file.exists():
            print(f"INFO: Loading cached chunks and catalog from {cache_file}")
            self.chunks, self.catalogs = torch.load(cache_file)
        else:
            self.chunks, self.catalogs = self._prerender_chunks(image)
            if cache_file is not None:
                print(f"INFO: Saving cached chunks and catalog to {cache_file}")
                torch.save((self.chunks, self.catalogs), cache_file)

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

    def _prerender_chunks(self, image):
        chunks = F.unfold(image, kernel_size=self.kernel_size, stride=self.stride)
        chunks = rearrange(
            chunks,
            "b (c h w) n -> (b n) c h w",
            c=image.shape[1],
            h=self.kernel_size,
            w=self.kernel_size,
        )
        catalogs = []
        chunks_with_galaxies = []
        encoder = self.encoder.to(self.prerender_device)
        with torch.no_grad():
            for chunk in tqdm(chunks):
                chunk_device = chunk.to(self.prerender_device)
                image_ptiles = encoder.get_images_in_ptiles(chunk_device.unsqueeze(0))
                tile_map = encoder.max_a_post(image_ptiles)
                if tile_map["galaxy_bool"].sum() > 0:
                    catalogs.append(cpu(tile_map))
                    chunks_with_galaxies.append(chunk.cpu())
        chunks_with_galaxies = torch.stack(chunks_with_galaxies, dim=0)
        # pylint: disable=consider-using-f-string
        print(
            "INFO: Number of chunks with galaxies: {ng}/{g}".format(
                ng=chunks_with_galaxies.shape[0], g=chunks.shape[0]
            )
        )
        return chunks_with_galaxies, catalogs

    def train_dataloader(self):
        return DataLoader(self, batch_size=2, num_workers=0, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self, batch_size=10, num_workers=0)

    def test_dataloader(self):
        return DataLoader(self, batch_size=1, num_workers=0)


def cpu(x: Dict[str, Tensor]):
    out = {}
    for k, v in x.items():
        out[k] = v.cpu()
    return out
