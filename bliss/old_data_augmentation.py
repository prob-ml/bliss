import random

import torch

from bliss.catalog import TileCatalog


def plocs_rot90(plocs, image_size):
    plocs_clone = plocs.clone()
    plocs_clone[..., 0], plocs_clone[..., 1] = image_size - plocs[..., 1] - 1, plocs[..., 0]
    return plocs_clone


def plocs_vflip(plocs, image_size):
    plocs_clone = plocs.clone()
    plocs_clone[..., 0] = image_size - plocs[..., 0] - 1
    return plocs_clone


class RotateFlipTransform(torch.nn.Module):
    def __init__(self, tile_slen, max_sources_per_tile):
        super().__init__()
        assert tile_slen % 2 == 0 and tile_slen > 1
        self.tile_slen = tile_slen
        self.max_sources_per_tile = max_sources_per_tile

    def __call__(self, datum, rotate_id=None, do_flip=None):
        # problematic if the psf isn't rotationally invariant
        datum_out = {"psf_params": datum["psf_params"]}

        d = {k: v.unsqueeze(0) for k, v in datum["tile_catalog"].items()}
        full_cat = TileCatalog(self.tile_slen, d).to_full_catalog()
        image_size = datum["images"].shape[1]

        # apply rotation
        if rotate_id is None:
            rotate_id = random.randint(0, 4)
        datum_out["images"] = datum["images"].rot90(rotate_id, [1, 2])
        datum_out["background"] = datum["background"].rot90(rotate_id, [1, 2])
        for _ in range(rotate_id):
            full_cat["plocs"] = plocs_rot90(full_cat["plocs"], image_size)

        # apply flip
        if do_flip is None:
            do_flip = random.choice([True, False])
        if do_flip:
            datum_out["images"] = datum_out["images"].flip([1])
            datum_out["background"] = datum_out["background"].flip([1])
            full_cat["plocs"] = plocs_vflip(full_cat["plocs"], image_size)

        aug_tile = full_cat.to_tile_catalog(self.tile_slen, self.max_sources_per_tile)
        d_out = {k: v.squeeze(0) for k, v in aug_tile.items()}
        datum_out["tile_catalog"] = d_out

        return datum_out
