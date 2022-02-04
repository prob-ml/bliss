#!/usr/bin/env python3
from os import getenv, environ
from pathlib import Path

import hydra

from bliss.train import train
from reconstruction import reconstruct

if not getenv("BLISS_HOME"):
    project_path = Path(__file__).resolve()
    bliss_home = project_path.parents[2]
    environ["BLISS_HOME"] = bliss_home.as_posix()


@hydra.main(config_path="./config", config_name="config")
def main(cfg):
    mode = cfg.mode
    if mode == "train":
        train(cfg)
    elif mode == "reconstruct":
        reconstruct(cfg)


if __name__ == "__main__":
    main()
