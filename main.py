#!/usr/bin/env python3

"""Main entry point(s) for BLISS."""
from os import environ, getenv
from pathlib import Path

import hydra

from bliss.generate import generate
from bliss.predict import predict
from bliss.train import train

if not getenv("BLISS_HOME"):
    project_path = Path(__file__).resolve()
    bliss_home = project_path.parents[0]
    environ["BLISS_HOME"] = bliss_home.as_posix()


# config_path should be overriden when running `bliss` poetry executable
# e.g., `bliss --cp case_studies/summer_template`
@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg):
    if cfg.mode == "generate":
        generate(cfg)
    elif cfg.mode == "train":
        train(cfg)
    elif cfg.mode == "predict":
        predict(cfg)
    else:
        raise KeyError


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter