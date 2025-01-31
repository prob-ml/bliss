
import logging
from pathlib import Path

import GCRCatalogs
import hydra
import numpy as np
import pandas as pd
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from bliss.surveys.dc2 import DC2DataModule

logging.basicConfig(level=logging.INFO)


# ------------- RAIL ------------ #
def create_rail_artifacts(rail_cfg: DictConfig):
    """Create DataFrames of ugrizy magnitudes and errors for RAIL training."""
    logging.info("Creating RAIL artifacts at %s", rail_cfg.processed_data_dir)
    log_dir = Path(rail_cfg.processed_data_dir)

    # Create output directory if it does not exist, or skip if artifacts already exist
    if not log_dir.exists():
        log_dir.mkdir(parents=True, exist_ok=True)
    elif not rail_cfg.pipeline.force_reprocess:
        logging.info("RAIL artifacts already exist. Skipping creation.")
        return

    lsst_root_dir = rail_cfg.pipeline.lsst_root_dir
    GCRCatalogs.set_root_dir(lsst_root_dir)
    lsst_catalog_gcr = GCRCatalogs.load_catalog(rail_cfg.pipeline.truth_match_catalog)
    lsst_catalog_subset = lsst_catalog_gcr.get_quantities(list(rail_cfg.pipeline.quantities))
    lsst_catalog_df = pd.DataFrame(lsst_catalog_subset)

    # Drop rows with inf or NaN
    lsst_catalog_df_na = lsst_catalog_df.replace([np.inf, -np.inf], np.nan)
    lsst_catalog_df_nona = lsst_catalog_df_na.dropna()

    # Rename some columns
    new_name = {
        "id_truth": "id",
        "mag_u_cModel": "mag_u_lsst",
        "mag_g_cModel": "mag_g_lsst",
        "mag_r_cModel": "mag_r_lsst",
        "mag_i_cModel": "mag_i_lsst",
        "mag_z_cModel": "mag_z_lsst",
        "mag_y_cModel": "mag_y_lsst",
        "magerr_u_cModel": "mag_err_u_lsst",
        "magerr_g_cModel": "mag_err_g_lsst",
        "magerr_r_cModel": "mag_err_r_lsst",
        "magerr_i_cModel": "mag_err_i_lsst",
        "magerr_z_cModel": "mag_err_z_lsst",
        "magerr_y_cModel": "mag_err_y_lsst",
    }

    lsst_catalog_df_nona_newname = lsst_catalog_df_nona.rename(new_name, axis=1)

    # Save pickle for RAIL-based training.
    train_nrow = rail_cfg.pipeline.train_size
    val_nrow = rail_cfg.pipeline.val_size
    lsst_catalog_df_nona_newname[:train_nrow].to_pickle(log_dir / "lsst_train_nona_200k.pkl")
    lsst_catalog_df_nona_newname[-1 - val_nrow - 1:].to_pickle(log_dir / "lsst_val_nona_100k.pkl")


# ------------- BLISS ----------- #
def create_bliss_artifacts(bliss_cfg: DictConfig):
    """CONSTRUCT BATCHES (.pt files) FOR DATA LOADING."""
    logging.info("Creating BLISS artifacts at %s", bliss_cfg.paths.processed_data_dir_bliss)
    dc2: DC2DataModule = instantiate(bliss_cfg.surveys.dc2)
    dc2.prepare_data()


@hydra.main(config_path="../", config_name="redshift")
def main(cfg: DictConfig) -> None:
    logging.info("Starting data generation")
    logging.info(OmegaConf.to_yaml(cfg))

    # Create RAIL artifacts
    create_rail_artifacts(cfg.rail)

    # Create BLISS artifacts
    create_bliss_artifacts(cfg)

    logging.info("Data generation complete")


if __name__ == "__main__":
    main()
