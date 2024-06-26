import random
import warnings
from pathlib import Path
from typing import TypedDict

import pytorch_lightning as pl
import torch
from torchdata.dataloader2 import (
    DataLoader2,
    DistributedReadingService,
    InProcessReadingService,
    MultiProcessingReadingService,
    SequentialReadingService,
)
from torchdata.datapipes import functional_datapipe
from torchdata.datapipes.iter import FileLister, IterableWrapper, IterDataPipe

from bliss.catalog import FullCatalog, TileCatalog

# prevent pytorch_lightning warning for num_workers = 2 in dataloaders with IterableDataset
warnings.filterwarnings(
    "ignore", ".*does not have many workers which may be a bottleneck.*", UserWarning
)
# an IterableDataset isn't supposed to have a __len__ method
warnings.filterwarnings("ignore", ".*Total length of .* across ranks is zero.*", UserWarning)


FileDatum = TypedDict(
    "FileDatum",
    {
        "tile_catalog": TileCatalog,
        "images": torch.Tensor,
        "background": torch.Tensor,
        "psf_params": torch.Tensor,
    },
)


@functional_datapipe("rotate_and_flip")
class RotateAndFlip(IterDataPipe):
    def __init__(self, datapipe):
        super().__init__()
        self.datapipe = datapipe

    def do_rotate_flip(self, datum_in, rotate_id, do_flip):
        # problematic if the psf isn't rotationally invariant
        datum_out = {"psf_params": datum_in["psf_params"]}

        # apply rotation
        datum_out["images"] = datum_in["images"].rot90(rotate_id, [1, 2])
        datum_out["background"] = datum_in["background"].rot90(rotate_id, [1, 2])
        d = datum_in["tile_catalog"]
        datum_out["tile_catalog"] = {k: v.rot90(rotate_id, [0, 1]) for k, v in d.items()}

        # apply flip
        if do_flip:
            datum_out["images"] = datum_out["images"].flip([1])
            datum_out["background"] = datum_out["background"].flip([1])
            for k, v in datum_out["tile_catalog"].items():
                datum_out["tile_catalog"][k] = v.flip([0])

        # locations require special logic
        if "locs" in datum_in["tile_catalog"]:
            locs = datum_out["tile_catalog"]["locs"]
            for _ in range(rotate_id):
                # Rotate 90 degrees clockwise (in pixel coordinates)
                locs = torch.stack((1 - locs[..., 1], locs[..., 0]), dim=3)
            if do_flip:
                locs = torch.stack((1 - locs[..., 0], locs[..., 1]), dim=3)
            datum_out["tile_catalog"]["locs"] = locs

        return datum_out

    def __iter__(self):
        for datum_in in self.datapipe:
            rotate_id = random.randint(0, 4)
            do_flip = random.choice([True, False])
            yield self.do_rotate_flip(datum_in, rotate_id, do_flip)


@functional_datapipe("random_shift")
class RandomShift(IterDataPipe):
    def __init__(self, datapipe, tile_slen, max_sources_per_tile):
        super().__init__()
        self.datapipe = datapipe
        assert tile_slen % 2 == 0 and tile_slen > 1
        self.tile_slen = tile_slen
        self.max_sources_per_tile = max_sources_per_tile

    def do_shift(self, datum_in, vertical_shift, horizontal_shift):
        datum_out = {"psf_params": datum_in["psf_params"]}

        for k in ("images", "background"):
            img = datum_in[k]
            img = torch.roll(img, shifts=vertical_shift, dims=1)
            img = torch.roll(img, shifts=horizontal_shift, dims=2)
            datum_out[k] = img

        d = {k: v.unsqueeze(0) for k, v in datum_in["tile_catalog"].items()}
        tile_cat = TileCatalog(self.tile_slen, d)
        full_cat = tile_cat.to_full_catalog()

        full_cat["plocs"][:, :, 0] += vertical_shift
        full_cat["plocs"][:, :, 1] += horizontal_shift

        aug_tile = full_cat.to_tile_catalog(
            self.tile_slen, self.max_sources_per_tile, filter_oob=True
        )
        d_out = {k: v.squeeze(0) for k, v in aug_tile.items()}
        datum_out["tile_catalog"] = d_out

        return datum_out

    def __iter__(self):
        for datum_in in self.datapipe:
            shift_ub = self.tile_slen // 2
            shift_lb = -(shift_ub - 1)
            vertical_shift = random.randint(shift_lb, shift_ub)
            horizontal_shift = random.randint(shift_lb, shift_ub)
            yield self.do_shift(datum_in, vertical_shift, horizontal_shift)


@functional_datapipe("full_catalog_to_tile")
class FullCatalogToTile(IterDataPipe):
    def __init__(self, datapipe, tile_slen, max_sources):
        super().__init__()

        self.datapipe = datapipe
        self.tile_slen = tile_slen
        self.max_sources = max_sources

    def __iter__(self):
        for datum_in in self.datapipe:
            datum_out = {k: v for k, v in datum_in.items() if k != "full_catalog"}

            h_pixels, w_pixels = datum_in["images"].shape[1:]
            full_cat = FullCatalog(h_pixels, w_pixels, datum_in["full_catalog"])
            tile_cat = full_cat.to_tile_catalog(self.tile_slen, self.max_sources).data
            d = {k: v.squeeze(0) for k, v in tile_cat.items()}
            datum_out["tile_catalog"] = d

            yield datum_out


@functional_datapipe("parse_pt")
class ReadTorchPt(IterDataPipe):
    def __init__(self, datapipe, shuffle=False) -> None:
        self.datapipe = datapipe
        self.shuffle = shuffle

    def __iter__(self):
        for filename in self.datapipe:
            file_datums = torch.load(filename)
            if self.shuffle:
                file_datums = IterableWrapper(file_datums).shuffle()
            yield from file_datums


class CachedSimulatedDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir,
        splits: str,
        split_seed: int = 0,
        batch_size: int = 32,
        num_workers: int = 0,
        convert_full_cat=False,
        tile_slen=None,
        max_sources=None,
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.splits = splits
        self.split_seed = split_seed
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.tile_slen = tile_slen
        self.max_sources_per_tile = max_sources
        self.convert_full_cat = convert_full_cat

        self.all_pipe = None
        self.train_pipe = None
        self.val_pipe = None
        self.test_pipe = None

        if self.num_workers == 0:
            mp_rs = InProcessReadingService()
        else:
            mp_rs = MultiProcessingReadingService(num_workers=self.num_workers)

        if torch.cuda.device_count() > 1:
            dist_rs = DistributedReadingService()
            self.reading_service = SequentialReadingService(dist_rs, mp_rs)
        else:
            self.reading_service = mp_rs

    def setup(self, stage=None):
        self.all_pipe = FileLister([self.data_dir]).filter(filter_fn=lambda f: f.endswith(".pt"))

        num_files = len(list(self.all_pipe))
        ds_names = ["train", "val", "test"]
        split_props = {k: float(v) for k, v in zip(ds_names, self.splits.split("/"))}

        self.train_pipe, self.val_pipe, self.test_pipe = self.all_pipe.random_split(
            total_length=num_files, weights=split_props, seed=self.split_seed
        )

    def train_dataloader(self) -> DataLoader2:
        pipe = self.train_pipe.shuffle().sharding_filter()  # shuffle files
        pipe = pipe.parse_pt(shuffle=True)
        if self.convert_full_cat:
            pipe = pipe.full_catalog_to_tile(self.tile_slen, self.max_sources_per_tile)
        pipe = pipe.rotate_and_flip()  # .random_shift(self.tile_slen, self.max_sources_per_tile)
        pipe = pipe.batch(self.batch_size).collate()
        return DataLoader2(pipe, reading_service=self.reading_service)

    def nontrain_dataloader(self, base_pipe) -> DataLoader2:
        pipe = base_pipe.sharding_filter().parse_pt()
        if self.convert_full_cat:
            pipe = pipe.full_catalog_to_tile(self.tile_slen, self.max_sources_per_tile)
        pipe = pipe.batch(self.batch_size).collate()
        return DataLoader2(pipe)

    def val_dataloader(self) -> DataLoader2:
        return self.nontrain_dataloader(self.val_pipe)

    def test_dataloader(self) -> DataLoader2:
        return self.nontrain_dataloader(self.test_pipe)

    def predict_dataloader(self) -> DataLoader2:
        return self.nontrain_dataloader(self.all_pipe)
