#!/usr/bin/env python3
from os import getenv, environ
from pathlib import Path

import hydra

if not getenv("BLISS_HOME"):
    project_path = Path(__file__).resolve()
    bliss_home = project_path.parents[2]
    environ["BLISS_HOME"] = bliss_home.as_posix()


@hydra.main(config_path="./config", config_name="config")
def main(cfg):
    if cfg.mode == "train":
        from bliss.train import train as task
    elif cfg.mode == "tune":
        from bliss.tune import tune as task
    elif cfg.mode == "generate":
        from bliss.generate import generate as task
    elif cfg.mode == "predict":
        from bliss.predict import predict as task
    else:
        raise KeyError
    task(cfg)


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
