import random

import torch


class LensingRotateFlipTransform(torch.nn.Module):
    def __init__(self, without_replacement=False):
        super().__init__()
        self.rotate_id = -1
        self.flip_id = -1
        self.seen_states = set()
        self.without_replacement = without_replacement

    def __call__(self, datum):
        self.rotate_id = random.randint(0, 3)
        self.flip_id = random.randint(0, 1)

        if self.without_replacement:
            if len(self.seen_states) == 8:
                self.seen_states = set()
            while ((self.rotate_id, self.flip_id)) in self.seen_states:
                self.rotate_id = random.randint(0, 3)
                self.flip_id = random.randint(0, 1)
            self.seen_states.add((self.rotate_id, self.flip_id))

        # problematic if the psf isn't rotationally invariant
        datum_out = {"psf_params": datum["psf_params"]}

        # apply rotation
        datum_out["images"] = datum["images"].rot90(self.rotate_id, [1, 2])
        d = datum["tile_catalog"]
        datum_out["tile_catalog"] = {k: v.rot90(self.rotate_id, [0, 1]) for k, v in d.items()}

        # apply flip
        if self.flip_id == 1:
            datum_out["images"] = datum_out["images"].flip([1])
            d = datum_out["tile_catalog"]
            datum_out["tile_catalog"] = {k: v.flip([0]) for k, v in d.items()}

        # shear requires special logic
        if all(k in datum["tile_catalog"] for k in ("shear_1", "shear_2")):
            shear1 = datum_out["tile_catalog"]["shear_1"]
            shear2 = datum_out["tile_catalog"]["shear_2"]
            for _ in range(self.rotate_id):
                shear1 = -shear1
                shear2 = -shear2
            if self.flip_id == 1:
                shear2 = -shear2
            datum_out["tile_catalog"]["shear_1"] = shear1
            datum_out["tile_catalog"]["shear_2"] = shear2

        # locations require special logic
        if "locs" in datum["tile_catalog"]:
            locs = datum_out["tile_catalog"]["locs"]
            for _ in range(self.rotate_id):
                # Rotate 90 degrees clockwise (in pixel coordinates)
                locs = torch.stack((1 - locs[..., 1], locs[..., 0]), dim=3)
            if self.flip_id == 1:
                locs = torch.stack((1 - locs[..., 0], locs[..., 1]), dim=3)
            datum_out["tile_catalog"]["locs"] = locs

        return datum_out
