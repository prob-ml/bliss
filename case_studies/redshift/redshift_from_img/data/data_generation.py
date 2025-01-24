from os import environ
from pathlib import Path

import GCRCatalogs
import pandas as pd
import torch
from hydra import compose, initialize
from hydra.utils import instantiate
import pickle
from bliss.catalog import SourceType
from bliss.surveys.dc2 import DC2DataModule, wcs_from_wcs_header_str


with initialize(config_path=".", version_base=None):
    notebook_cfg = compose("data")

print("initialization ends", flush=True)
print("+" * 100, flush=True)

print("+" * 100, flush=True)
print("load dc2", flush=True)
dc2: DC2DataModule = instantiate(notebook_cfg.surveys.dc2)
dc2.prepare_data()
print("+" * 100, flush=True)