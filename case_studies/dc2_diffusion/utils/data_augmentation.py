import random

import torch

from bliss.catalog import TileCatalog

class HalfPixelRandomShiftTransform(torch.nn.Module):
    def __init__(self, tile_slen, max_sources_per_tile):
        super().__init__()
        assert tile_slen == 0.5
        self.tile_slen = tile_slen
        self.max_sources_per_tile = max_sources_per_tile

    def __call__(self, datum, vertical_shift=None, horizontal_shift=None):
        datum_out = {"psf_params": datum["psf_params"]}

        shift_ub = 2
        shift_lb = -1
        if vertical_shift is None:
            vertical_shift = random.randint(shift_lb, shift_ub)
        if horizontal_shift is None:
            horizontal_shift = random.randint(shift_lb, shift_ub)

        img = datum["images"]
        img = torch.roll(img, shifts=vertical_shift, dims=1)
        img = torch.roll(img, shifts=horizontal_shift, dims=2)
        datum_out["images"] = img

        d = {k: v.unsqueeze(0) for k, v in datum["tile_catalog"].items()}
        tile_cat = TileCatalog(d)
        full_cat = tile_cat.to_full_catalog(self.tile_slen)

        full_cat["plocs"][:, :, 0] += vertical_shift
        full_cat["plocs"][:, :, 1] += horizontal_shift

        aug_tile = full_cat.to_tile_catalog(
            self.tile_slen, self.max_sources_per_tile, filter_oob=True
        )
        d_out = {k: v.squeeze(0) for k, v in aug_tile.items()}
        datum_out["tile_catalog"] = d_out

        return datum_out