import logging
from pathlib import Path

import numpy as np
import rail  # pylint: disable=import-error
from hydra import compose, initialize
from rail.core.stage import RailStage  # pylint: disable=import-error
from rail.estimation.algos.flexzboost import FlexZBoostInformer  # pylint: disable=import-error

logging.basicConfig(level=logging.INFO)

RailStage.data_store.__class__.allow_overwrite = True

# %% configs
with initialize(config_path="../", version_base=None):
    notebook_cfg = compose("redshift_flexzboost")
rail_dir = Path(notebook_cfg.paths["processed_data_dir_rail"])

# parameters for optimization
fz_dict = {
    "zmin": 0.0,
    "zmax": 3.0,
    "nzbins": 301,
    "trainfrac": 0.75,
    "bumpmin": 0.02,
    "bumpmax": 0.35,
    "nbump": 20,
    "sharpmin": 0.7,
    "sharpmax": 2.1,
    "nsharp": 15,
    "max_basis": 35,
    "basis_system": "cosine",
    "hdf5_groupname": None,
    "nondetect_val": np.nan,
    "regression_params": {"max_depth": 8, "objective": "reg:squarederror"},
}

# %% get all filenames like rail_dir / train_df_{i}
# listing all files in the directory
train_df_files = list(rail_dir.glob("train_df_*.hdf5"))
# sorting the files by their index
train_df_files.sort(key=lambda x: int(x.stem.split("_")[-1]))

# %% load training data
for i, train_df_file in enumerate(train_df_files):
    out_model_fn = rail_dir / f"flexzboost_model_results_{i}.pkl"

    logging.info(f"Trainset {i} from file {train_df_file} --> {out_model_fn}")

    training_data = RailStage.data_store.read_file(
        f"training_data_{i}",
        rail.core.data.Hdf5Handle,
        train_df_file,
    )

    inform_pzflex = FlexZBoostInformer.make_stage(
        name="inform_fzboost", model=out_model_fn, **fz_dict
    )
    inform_pzflex.inform(training_data)
