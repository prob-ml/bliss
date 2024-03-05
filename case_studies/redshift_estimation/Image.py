import sys
import os
os.chdir('/home/yolandz/bliss')

from bliss.encoder.variational_dist import VariationalDistSpec, VariationalDist
from bliss.encoder.unconstrained_dists import UnconstrainedNormal
import torch
import numpy as np
from os import environ
from pathlib import Path
from hydra import initialize, compose
from hydra.utils import instantiate
import matplotlib.pyplot as plt
from omegaconf import DictConfig, OmegaConf
from bliss.catalog import TileCatalog

environ["BLISS_HOME"] = "/home/yolandz/bliss"
with initialize(config_path=".", version_base=None):
    cfg = compose("redshift", overrides={"surveys.sdss.load_image_data=true"})

simulated_dataset = instantiate(cfg.generate.simulator, num_workers=0)
simulated_batch_of_data=simulated_dataset.get_batch()
ims = simulated_batch_of_data['images']
plt.imshow(ims[0][0], vmax = 500, vmin = 100)
plt.imshow(ims[0][2], vmax = 500, vmin = 100)