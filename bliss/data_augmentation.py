import random

import torch

from bliss.catalog import TileCatalog


class RotateFlipTransform(torch.nn.Module):
    def __call__(self, datum, rotate_id=None, do_flip=None):
        # problematic if the psf isn't rotationally invariant
        datum_out = {"psf_params": datum["psf_params"]}

        # apply rotation
        if rotate_id is None:
            rotate_id = random.randint(0, 4)
        datum_out["images"] = datum["images"].rot90(rotate_id, [1, 2])
        d = datum["tile_catalog"]
        datum_out["tile_catalog"] = {k: v.rot90(rotate_id, [0, 1]) for k, v in d.items()}

        # apply flip
        if do_flip is None:
            do_flip = random.choice([True, False])
        if do_flip:
            datum_out["images"] = datum_out["images"].flip([1])
            for k, v in datum_out["tile_catalog"].items():
                datum_out["tile_catalog"][k] = v.flip([0])

        # locations require special logic
        if "locs" in datum["tile_catalog"]:
            locs = datum_out["tile_catalog"]["locs"]
            for _ in range(rotate_id):
                # Rotate 90 degrees clockwise (in pixel coordinates)
                locs = torch.stack((1 - locs[..., 1], locs[..., 0]), dim=3)
            if do_flip:
                locs = torch.stack((1 - locs[..., 0], locs[..., 1]), dim=3)
            datum_out["tile_catalog"]["locs"] = locs

        return datum_out


class RandomShiftTransform(torch.nn.Module):
    def __init__(self, tile_slen, max_sources_per_tile):
        super().__init__()
        assert tile_slen % 2 == 0 and tile_slen > 1
        self.tile_slen = tile_slen
        self.max_sources_per_tile = max_sources_per_tile

    def __call__(self, datum, vertical_shift=None, horizontal_shift=None):
        datum_out = {"psf_params": datum["psf_params"]}

        shift_ub = self.tile_slen // 2
        shift_lb = -(shift_ub - 1)
        if vertical_shift is None:
            vertical_shift = random.randint(shift_lb, shift_ub)
        if horizontal_shift is None:
            horizontal_shift = random.randint(shift_lb, shift_ub)

        img = datum["images"]
        img = torch.roll(img, shifts=vertical_shift, dims=1)
        img = torch.roll(img, shifts=horizontal_shift, dims=2)
        datum_out["images"] = img

        tile_cat = TileCatalog.from_dict(datum["tile_catalog"])
        full_cat = tile_cat.to_full_catalog(self.tile_slen)

        full_cat["plocs"][:, :, 0] += vertical_shift
        full_cat["plocs"][:, :, 1] += horizontal_shift

        aug_tile = full_cat.to_tile_catalog(
            self.tile_slen, self.max_sources_per_tile, filter_oob=True
        )
        datum_out["tile_catalog"] = aug_tile.to_dict()

        return datum_out
