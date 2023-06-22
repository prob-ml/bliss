#!/usr/bin/env python3

"""Main entry point(s) for BLISS."""
import logging
from os import environ, getenv
from pathlib import Path

import hydra

from bliss.api import BlissClient

if not getenv("BLISS_HOME"):
    project_path = Path(__file__).resolve()
    bliss_home = project_path.parents[1]
    environ["BLISS_HOME"] = bliss_home.as_posix()

    logger = logging.getLogger(__name__)
    logger.warning(
        "WARNING: BLISS_HOME not set, setting to project root %s\n",  # noqa: WPS323
        environ["BLISS_HOME"],
    )


# config_path should be overriden when running `bliss` poetry executable
# e.g., `bliss -cp case_studies/summer_template -cn config`
@hydra.main(config_path="conf", config_name="base_config", version_base=None)
def main(cfg):
    bliss_client = BlissClient(cwd=cfg.paths.root)
    if cfg.mode == "generate":
        bliss_client.generate(
            n_batches=cfg.generate.n_batches,
            batch_size=cfg.generate.batch_size,
            max_images_per_file=cfg.generate.max_images_per_file,
            **cfg,
        )
    elif cfg.mode == "train":
        bliss_client.train(weight_save_path=cfg.training.weight_save_path, **cfg)
    elif cfg.mode == "predict":
        bliss_client.predict_sdss(weight_save_path=cfg.predict.weight_save_path, **cfg)
    else:
        raise KeyError


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
