#!/usr/bin/env python3
from os import environ, getenv
from pathlib import Path

import hydra

from bliss.generate import generate
from bliss.train import train

if not getenv("BLISS_HOME"):
    project_path = Path(__file__).resolve()
    bliss_home = project_path.parents[2]
    environ["BLISS_HOME"] = bliss_home.as_posix()


@hydra.main(config_path="./config", config_name="config")
def main(cfg):
    if cfg.mode == "train":
        train(cfg)
    elif cfg.mode == "generate":
        filepath = cfg.generate.file + ".pt"
        imagepath = cfg.generate.file + ".png"
        generate(cfg.generate.dataset, filepath, imagepath, cfg.generate.n_plots)
    else:
        raise KeyError


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
