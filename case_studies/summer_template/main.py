#!/usr/bin/env python3
from os import environ, getenv
from pathlib import Path

import hydra
import torch
from hydra.utils import instantiate

from bliss.train import train

if not getenv("BLISS_HOME"):
    project_path = Path(__file__).resolve()
    bliss_home = project_path.parents[2]
    environ["BLISS_HOME"] = bliss_home.as_posix()


def prepare_image(x):
    x = torch.from_numpy(x).cuda().unsqueeze(0)
    return x[:, :, :1488, :720]


def predict(cfg):
    sdss = instantiate(cfg.inference.dataset)
    encoder = instantiate(cfg.encoder).cuda()
    # TODO: load saved weights
    encoder.eval()
    batch = {
        "images": prepare_image(sdss[0]["image"]),
        "background": prepare_image(sdss[0]["background"]),
    }
    pred = encoder.encode_batch(batch)
    est_cat = encoder.variational_mode(pred)
    print(est_cat)


@hydra.main(config_path=".", config_name="config", version_base=None)
def main(cfg):
    if cfg.mode == "train":
        train(cfg)
    elif cfg.mode == "predict":
        predict(cfg)
    else:
        raise KeyError


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
