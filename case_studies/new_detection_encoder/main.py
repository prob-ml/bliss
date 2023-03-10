#!/usr/bin/env python3
from os import environ, getenv
from pathlib import Path

import hydra
from einops import rearrange
from omegaconf import OmegaConf
from yolov5.models.yolo import DetectionModel

from bliss.models.detection_encoder import DetectionEncoder
from bliss.train import train


class NewDetectionEncoder(DetectionEncoder):
    def __init__(self, **args):
        architecture = args.pop("architecture")
        super().__init__(**args)

        arch_dict = OmegaConf.to_container(architecture)
        self.model = DetectionModel(cfg=arch_dict, ch=2)

        self.tiles_to_crop = (args["ptile_slen"] - args["tile_slen"]) // (2 * args["tile_slen"])

    def do_encode_batch(self, images_with_background):
        # setting this to true every time is a hack to make yolo DetectionModel
        # give us output of the right dimension
        self.model.model[-1].training = True

        output = self.model(images_with_background)

        ttc = self.tiles_to_crop
        output_cropped = output[0][:, :, ttc:-ttc, ttc:-ttc, :]
        return rearrange(output_cropped, "b c h w p -> (b c h w) p")


@hydra.main(config_path="./config", config_name="config", version_base=None)
def main(cfg):
    train(cfg)


if not getenv("BLISS_HOME"):
    project_path = Path(__file__).resolve()
    bliss_home = project_path.parents[2]
    environ["BLISS_HOME"] = bliss_home.as_posix()

if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
