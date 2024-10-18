import torch
import random
    

class RandomRotate90DegreesClockwiseFlipTransformCombined(torch.nn.Module):
    def __init__(self, with_replacement=False):
        super().__init__()
        self.rotate_action = -1
        self.flip_action = -1
        self.seen_states = set()
        self.with_replacement = with_replacement

    def __call__(self, datum):
        self.rotate_action = random.randint(0, 3)
        self.flip_action = random.randint(0, 1)
        if self.with_replacement:
            if len(self.seen_states) == 8:
                self.seen_states = set()
            while ((self.rotate_action, self.flip_action)) in self.seen_states:
                self.rotate_action = random.randint(0, 3)
                self.flip_action = random.randint(0, 1)
            self.seen_states.add((self.rotate_action, self.flip_action))
        
        datum_out = {"psf_params": datum["psf_params"]}

    # apply rotation
        datum_out["images"] = datum["images"].rot90(self.rotate_action, [1, 2])
        d = datum["tile_catalog"]
        datum_out["tile_catalog"] = {k: v.rot90(self.rotate_action, [0, 1]) for k, v in d.items()}

        # locations require special logic
        if "locs" in datum["tile_catalog"]:
            locs = datum_out["tile_catalog"]["locs"]
            for _ in range(self.rotate_action):
                # Rotate 90 degrees clockwise (in pixel coordinates)
                locs = torch.stack((1 - locs[..., 1], locs[..., 0]), dim=3)

        shear1, shear2 = None, None
        if "shear_1" in datum["tile_catalog"]:
            shear1 = datum_out["tile_catalog"]["shear_1"]
        if "shear_2" in datum["tile_catalog"]:
            shear2 = datum_out["tile_catalog"]["shear_2"]
            for _ in range(self.rotate_action):
                if shear1 is not None:
                    shear1 = -shear1
                if shear2 is not None:
                    shear2 = -shear2
        datum_out["tile_catalog"]["shear_1"] = shear1
        datum_out["tile_catalog"]["shear_2"] = shear2
        
        if self.flip_action == 1:
            # apply flip
            datum_out["images"] = datum_out["images"].flip([1])
            d = datum_out["tile_catalog"]
            datum_out["tile_catalog"] = {k: v.flip([0]) for k, v in d.items()}

            # locations require special logic
            if "locs" in datum_out["tile_catalog"]:
                locs = datum_out["tile_catalog"]["locs"]
                locs = torch.stack((1 - locs[..., 0], locs[..., 1]), dim=3)
                datum_out["tile_catalog"]["locs"] = locs

            if "shear_2" in datum_out["tile_catalog"]:
                shear2 = datum_out["tile_catalog"]["shear_2"]
                shear2 = -shear2
                datum_out["tile_catalog"]["shear_2"] = shear2

        return datum_out