import sys
import copy

sys.path.append("../../")

from einops import rearrange
from bliss.catalog import TileCatalog, get_images_in_tiles, get_is_on_from_n_sources
from bliss import reporting
from bliss.encoder import Encoder
from bliss.inference import SDSSFrame
from bliss.datasets import sdss
from bliss.inference import reconstruct_scene_at_coordinates
from case_studies.strong_lensing.plots.main import load_models

import matplotlib.pyplot as plt

plt.style.use("ggplot")
import torch

from astropy.table import Table
import plotly.express as px
import plotly.graph_objects as go
from hydra import compose, initialize
from hydra.utils import instantiate
import numpy as np

with initialize(config_path="config"):
    cfg = compose("config", overrides=[])

device = torch.device("cuda:0")
enc, dec = load_models(cfg, device)
bp = enc.border_padding
torch.cuda.empty_cache()

enc.n_rows_per_batch = 10
enc.n_images_per_batch = 15

dataset = instantiate(
    cfg.datasets.simulated,
    generate_device="cuda:0",
)

test_type = "galaxy"

if test_type == "lens":
    total_contained = np.zeros(12)
else:
    total_contained = np.zeros(7)

total_lenses = 0
batch_size = 1
trials = 10_000
num_samples = 100

for _ in range(trials):
    tile_catalog = dataset.sample_prior(
        batch_size, cfg.datasets.simulated.n_tiles_h, cfg.datasets.simulated.n_tiles_w
    )
    tile_catalog.set_all_fluxes_and_mags(dataset.image_decoder)
    images, backgrounds = dataset.simulate_image_from_catalog(tile_catalog)

    full_true = tile_catalog.cpu().to_full_params()

    if test_type == "lens":
        true_gal_bool = full_true["lensed_galaxy_bools"].cpu().numpy()[0, :, 0].astype("bool")
        true_lens = full_true["lens_params"].cpu().numpy()[0, ...][true_gal_bool]
    else:
        true_gal_bool = full_true["galaxy_bools"].cpu().numpy()[0, :, 0].astype("bool")
        true_lens = full_true["galaxy_params"].cpu().numpy()[0, ...][true_gal_bool]

    samples = []
    for _ in range(num_samples):
        tile_sample_dict = enc.sample(images, backgrounds, 1, tile_catalog)
        tile_sample = TileCatalog.from_flat_dict(
            enc.detection_encoder.tile_slen,
            cfg.datasets.simulated.n_tiles_h,
            cfg.datasets.simulated.n_tiles_w,
            {k: v.squeeze(0) for k, v in tile_sample_dict.items()},
        )
        tile_sample.locs = copy.deepcopy(tile_catalog.locs)
        tile_sample.n_sources = copy.deepcopy(tile_catalog.n_sources)
        full_sample = tile_sample.cpu().to_full_params()
        if test_type == "lens":
            lens_sample = full_sample["lens_params"].cpu().numpy()[0, ...][true_gal_bool]
        else:
            lens_sample = full_sample["galaxy_params"].cpu().numpy()[0, ...][true_gal_bool]
        samples.append(lens_sample)

    samples = np.array(samples)

    confidence_percent = 0.90
    alpha = ((1 - confidence_percent) / 2) * 100
    lower_ci = np.percentile(samples, alpha, axis=0)
    upper_ci = np.percentile(samples, 100 - alpha, axis=0)

    contained = np.logical_and(lower_ci <= true_lens, true_lens <= upper_ci).astype("int")
    total_contained += np.sum(contained, axis=0).squeeze()
    total_lenses += true_lens.shape[0]

print(total_contained / total_lenses)
