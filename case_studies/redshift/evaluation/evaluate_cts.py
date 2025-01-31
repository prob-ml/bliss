
import os
os.environ['OMP_NUM_THREADS'] = '16'
os.environ['MKL_NUM_THREADS'] = '16'
os.environ['NUMEXPR_NUM_THREADS'] = '16'
from os import environ
from pathlib import Path
import torch
from einops import rearrange
import pickle
from tqdm import tqdm
import numpy as np
import hydra
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from omegaconf import DictConfig
from hydra import initialize, compose
from hydra.utils import instantiate

from pytorch_lightning.utilities import move_data_to_device

from bliss.catalog import FullCatalog, BaseTileCatalog, TileCatalog
from bliss.surveys.dc2 import DC2DataModule
from case_studies.redshift.evaluation.utils.load_lsst import get_lsst_full_cat
from case_studies.redshift.evaluation.utils.safe_metric_collection import SafeMetricCollection as MetricCollection
from case_studies.redshift.redshift_from_img.encoder.metrics import RedshiftMeanSquaredErrorBin

def get_best_ckpt(ckpt_dir: str):
    ckpt_dir = Path(ckpt_dir)
    ckpt_files = list(ckpt_dir.glob("*.ckpt"))
    sorted_files = sorted(ckpt_files, key=lambda f: float(f.stem.split('_')[1]))
    if sorted_files:
        return sorted_files[0]
    else:
        raise FileExistsError("No ckpt files found in the directory")

@hydra.main(config_path=".", config_name="continuous_eval")
def main(cfg: DictConfig):
    # with initialize(config_path=".", job_name="continuous_eval"):
    #     cfg = compose(config_name="continuous_eval")
    output_dir = cfg.paths.plot_dir
    ckpt_dir = cfg.paths.ckpt_dir

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = get_best_ckpt(ckpt_dir)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # set up testing dataset
    dataset = instantiate(cfg.train.data_source)
    dataset.setup("test")

    # load bliss trained model - continuous version
    bliss_encoder = instantiate(cfg.encoder).to(device=device)
    pretrained_weights = torch.load(ckpt_path, device)["state_dict"]
    bliss_encoder.load_state_dict(pretrained_weights)
    bliss_encoder.eval()

    # load bliss trained model - continuous version
    bliss_output_path = output_dir / "cts_mode_metrics.pkl"

    # compute metrics -- continuous version
    if not bliss_output_path.exists():
        for batch_idx, batch in tqdm(enumerate(dataset.test_dataloader()), total=len(dataset.test_dataloader())):
            batch["images"] = batch["images"].to(device)
            bliss_encoder.update_metrics(batch, batch_idx)
        bliss_out_dict = bliss_encoder.mode_metrics.compute()

        with open(bliss_output_path, "wb") as outp:  # Overwrites any existing file.
            pickle.dump(bliss_out_dict, outp, pickle.HIGHEST_PROTOCOL)

    return

if __name__ == "__main__":
    main()



  
