#!/usr/bin/env python3
from os import environ, getenv
from pathlib import Path

import hydra

from bliss.generate import generate
from bliss.predict import predict
from bliss.train import train

if not getenv("BLISS_HOME"):
    project_path = Path(__file__).resolve()
    bliss_home = project_path.parents[2]
    environ["BLISS_HOME"] = bliss_home.as_posix()


@hydra.main(config_path=".", config_name="config", version_base=None)
def main(cfg):
    if cfg.mode == "train":
        train(cfg)
    elif cfg.mode == "predict":
        predict(cfg)
    elif cfg.mode == "generate":
        generate(cfg)
    else:
        raise KeyError


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
