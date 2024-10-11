import torch


class Rotate90DegreesClockwiseTransform(torch.nn.Module):
    def __init__(self, num_rotates):
        super().__init__()
        self.num_rotates = num_rotates

    def __call__(self, datum):
        # num_rotates in {1,2,3} - 90, 180, 270 degrees
        datum_out = {"psf_params": datum["psf_params"]}

        # apply rotation
        datum_out["images"] = datum["images"].rot90(self.num_rotates, [1, 2])
        d = datum["tile_catalog"]
        datum_out["tile_catalog"] = {k: v.rot90(self.num_rotates, [0, 1]) for k, v in d.items()}

        # locations require special logic
        if "locs" in datum["tile_catalog"]:
            locs = datum_out["tile_catalog"]["locs"]
            for _ in range(self.num_rotates):
                # Rotate 90 degrees clockwise (in pixel coordinates)
                locs = torch.stack((1 - locs[..., 1], locs[..., 0]), dim=3)

        shear1, shear2 = None, None
        if "shear_1" in datum["tile_catalog"]:
            shear1 = datum_out["tile_catalog"]["shear_1"]
        if "shear_2" in datum["tile_catalog"]:
            shear2 = datum_out["tile_catalog"]["shear_2"]
            for _ in range(self.num_rotates):
                if shear1 is not None:
                    shear1 = -shear1
                if shear2 is not None:
                    shear2 = -shear2
        datum_out["tile_catalog"]["shear_1"] = shear1
        datum_out["tile_catalog"]["shear_2"] = shear2
        return datum_out


class HorizontalFlipTransform(torch.nn.Module):
    def __call__(self, datum):
        # problematic if the psf isn't rotationally invariant
        datum_out = {"psf_params": datum["psf_params"]}

        # apply flip
        datum_out["images"] = datum["images"].flip([1])
        d = datum["tile_catalog"]
        datum_out["tile_catalog"] = {k: v.flip([0]) for k, v in d.items()}

        # locations require special logic
        if "locs" in datum["tile_catalog"]:
            locs = datum_out["tile_catalog"]["locs"]
            locs = torch.stack((1 - locs[..., 0], locs[..., 1]), dim=3)
            datum_out["tile_catalog"]["locs"] = locs

        if "shear_2" in datum["tile_catalog"]:
            shear2 = datum_out["tile_catalog"]["shear_2"]
            shear2 = -shear2
            datum_out["tile_catalog"]["shear_2"] = shear2

        return datum_out
