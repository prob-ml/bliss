import logging

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from case_studies.redshift.artifacts.redshift_dc2 import RedshiftDC2DataModule

logging.basicConfig(level=logging.INFO)


# ------------- BLISS ----------- #
def create_bliss_artifacts(bliss_cfg: DictConfig):
    """CONSTRUCT BATCHES (.pt files) FOR DATA LOADING."""
    logging.info("Creating BLISS artifacts at %s", bliss_cfg.paths.processed_data_dir_bliss)
    dc2: RedshiftDC2DataModule = instantiate(bliss_cfg.surveys.dc2)
    dc2.prepare_data()


@hydra.main(config_path="../", config_name="redshift")
def main(cfg: DictConfig) -> None:
    logging.info("Starting data generation")
    logging.info(OmegaConf.to_yaml(cfg))

    # Create BLISS artifacts
    create_bliss_artifacts(cfg)

    logging.info("Data generation complete")


if __name__ == "__main__":
    main()
