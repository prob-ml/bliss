from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pytorch_lightning as pl
import torch
from einops.einops import rearrange
from torch import Tensor
from torch.nn import functional as F
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from bliss.datasets.sdss import SloanDigitalSkySurvey
from bliss.encoder import Encoder
from bliss.models.binary import BinaryEncoder
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
        )
        image = torch.from_numpy(sdss_data[0]["image"][0])
        image = rearrange(image, "h w -> 1 1 h w")
        background = torch.from_numpy(sdss_data[0]["background"][0])
        background = rearrange(background, "h w -> 1 1 h w")

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
        self.chunks, self.bgs, self.catalogs = self._prerender_chunks(image, background)

    def __len__(self):
        return len(self.catalogs)

    def __getitem__(self, idx):
        chunk = self.chunks[idx]
        bg = self.bgs[idx]
        tile_map = self.catalogs[idx]
        return {
            "images": chunk.unsqueeze(0),
            "background": bg.unsqueeze(0),
            **tile_map,
            "slen": torch.tensor([self.slen]).unsqueeze(0),
        }

    @staticmethod
    def _collate(tile_catalogs: List[Dict[str, Tensor]]):
        out = {}
        for k in tile_catalogs[0]:
            out[k] = torch.cat([x[k] for x in tile_catalogs], dim=0)
        return out

    def _prerender_chunks(self, image, background):
        chunks = make_image_into_chunks(image, self.kernel_size, self.stride)
        bg_chunks = make_image_into_chunks(background, self.kernel_size, self.stride)
        catalogs = []
        chunks_with_galaxies = []
        bgs_with_galaxies = []
        encoder = self.encoder.to(self.prerender_device)
        with torch.no_grad():
            for chunk, bg in tqdm(zip(chunks, bg_chunks)):
                chunk_device = chunk.to(self.prerender_device).unsqueeze(0)
                bg_device = bg.to(self.prerender_device).unsqueeze(0)
                tile_map = encoder.max_a_post(chunk_device, bg_device)
                if tile_map["galaxy_bools"].sum() > 0:
                    catalogs.append(cpu(tile_map))
                    chunks_with_galaxies.append(chunk.cpu())
                    bgs_with_galaxies.append(bg.cpu())
        chunks_with_galaxies = torch.stack(chunks_with_galaxies, dim=0)
        bgs_with_galaxies = torch.stack(bgs_with_galaxies, dim=0)
        # pylint: disable=consider-using-f-string
        msg = "INFO: Number of chunks with galaxies: {ng}/{g}".format(
            ng=chunks_with_galaxies.shape[0],
            g=chunks.shape[0],
        )
        print(msg)
        return chunks_with_galaxies, bgs_with_galaxies, catalogs

    def train_dataloader(self):
        return DataLoader(self, batch_size=2, num_workers=0, shuffle=True, collate_fn=self._collate)

    def val_dataloader(self):
        return DataLoader(self, batch_size=10, num_workers=0, collate_fn=self._collate)

    def test_dataloader(self):
        return DataLoader(self, batch_size=1, num_workers=0, collate_fn=self._collate)


def make_image_into_chunks(image, kernel_size, stride):
    chunks = F.unfold(image, kernel_size=kernel_size, stride=stride)
    return rearrange(
        chunks,
        "b (c h w) n -> (b n) c h w",
        c=image.shape[1],
        h=kernel_size,
        w=kernel_size,
    )


def cpu(x: Dict[str, Tensor]):
    out = {}
    for k, v in x.items():
        out[k] = v.cpu()
    return out
