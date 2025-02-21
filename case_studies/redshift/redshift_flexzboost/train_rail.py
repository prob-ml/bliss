from pathlib import Path

import numpy as np
import rail  # pylint: disable=import-error
from hydra import compose, initialize
from rail.core.stage import RailStage  # pylint: disable=import-error
from rail.estimation.algos.flexzboost import FlexZBoostInformer  # pylint: disable=import-error

RailStage.data_store.__class__.allow_overwrite = True

# %% configs
with initialize(config_path="../../", version_base=None):
    notebook_cfg = compose("redshift")
rail_dir = Path(notebook_cfg.paths["processed_data_dir_rail"])
out_model_fn = rail_dir / "flexzboost_model_results.pkl"

# parameters foroptimization
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

# %% load training data
training_data = RailStage.data_store.read_file(
    "training_data", rail.core.data.Hdf5Handle, rail_dir / "train_df.hdf5"
)

# %% do the training
inform_pzflex = FlexZBoostInformer.make_stage(name="inform_fzboost", model=out_model_fn, **fz_dict)
inform_pzflex.inform(training_data)
