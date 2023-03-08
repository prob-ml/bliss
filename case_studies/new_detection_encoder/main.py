#!/usr/bin/env python3
from os import environ, getenv
from pathlib import Path

import hydra
import numpy as np
import torch
from einops import rearrange
from omegaconf import OmegaConf
from yolov5.models.yolo import DetectionModel

from bliss.models.detection_encoder import DetectionEncoder
from bliss.train import train


class NewDetectionEncoder(DetectionEncoder):
    def __init__(self, **ode_args):
        architecture = ode_args.pop("architecture")
        super().__init__(**ode_args)

        # the checkerboard sizes below only make sense for images composed of an
        # 8x8 grid of 4x4 tiles. It's straightforward to generalize it to arbitrary
        # image/tile sizes, but I done so because I haven't determined yet whether
        # this checkerboard is helpful/necessary
        shape = (20, 20)
        base_checkerboard = np.indices(shape).sum(axis=0) % 2
        self.checkerboard = torch.from_numpy(base_checkerboard).unsqueeze(2).repeat([1, 4, 4])
        self.checkerboard = self.checkerboard.view(1, 1, 80, 80).float().cuda()

        arch_dict = OmegaConf.to_container(architecture)
        self.model = DetectionModel(cfg=arch_dict, ch=3)

    def do_encode_batch(self, images_with_background):
        b = images_with_background.size(0)
        batch_checkerboard = self.checkerboard.expand([b, 1, -1, -1])
        x = torch.cat([images_with_background, batch_checkerboard], dim=1)
        self.model.model[-1].training = True
        output = self.model(x)
        output_cropped = output[0][:, :, 6:-6, 6:-6, :]
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
