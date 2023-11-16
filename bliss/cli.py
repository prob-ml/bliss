import logging
from os import environ, getenv
from pathlib import Path

import hydra

from bliss.generate import generate
from bliss.predict import predict
from bliss.train import train

# pragma: no cover
# ============================== CLI ==============================


# config_path should be overriden when running `bliss` poetry executable
# e.g., `bliss -cp case_studies/summer_template -cn config`
@hydra.main(config_path="conf", config_name="base_config", version_base=None)
def main(cfg):
    """Main entry point(s) for BLISS."""
    if not getenv("BLISS_HOME"):
        project_path = Path(__file__).resolve()
        bliss_home = project_path.parents[1]
        environ["BLISS_HOME"] = bliss_home.as_posix()

        logger = logging.getLogger(__name__)
        logger.warning(
            "WARNING: BLISS_HOME not set, setting to project root %s\n",  # noqa: WPS323
            environ["BLISS_HOME"],
        )

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

# pragma: no cover
