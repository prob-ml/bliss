import sys

sys.path.append("../../")

import matplotlib.pyplot as plt
from einops import rearrange

from bliss import metrics
from bliss.catalog import TileCatalog, get_images_in_tiles, get_is_on_from_n_sources
from bliss.datasets import sdss
from bliss.encoder import Encoder
from bliss.inference import SDSSFrame, reconstruct_scene_at_coordinates
from case_studies.strong_lensing.plots.main import load_models

plt.style.use("ggplot")
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import torch
from astropy.table import Table
from hydra import compose, initialize
from hydra.utils import instantiate

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

delta = 0.025
increments = np.arange(0, 1.0, delta)
galaxies = np.zeros(increments.shape)
counts = np.zeros(increments.shape)

batch_size = 8
trials = 1000
for _ in range(trials):
    tile_catalog = dataset.sample_prior(
        batch_size, cfg.datasets.simulated.n_tiles_h, cfg.datasets.simulated.n_tiles_w
    )
    tile_catalog.set_all_fluxes_and_mags(dataset.image_decoder)
    images, backgrounds = dataset.simulate_image_from_catalog(tile_catalog)

    tile_map = enc.variational_mode(images, backgrounds, tile_catalog)
    full_map = tile_map.cpu().to_full_params()
    full_true = tile_catalog.cpu().to_full_params()

    for i in range(batch_size):
        true_gal_bool = full_true["lensed_galaxy_bools"].cpu().numpy()[i, :, 0].astype("bool")
        pred_gal_probs = full_map["lensed_galaxy_probs"].cpu().numpy()[i, ...]

        gal_probs = pred_gal_probs[true_gal_bool]
        gal_buckets = (gal_probs // delta).astype("int")
        for bucket in gal_buckets:
            galaxies[bucket] += 1

        count_buckets = (pred_gal_probs // delta).astype("int")
        for bucket in count_buckets:
            counts[bucket] += 1

x = []
y = []
for i, (galaxy, count) in enumerate(zip(galaxies, counts)):
    if count > 0:
        x.append(increments[i])
        y.append(galaxy / count)

plt.plot(increments, increments, color="b")
plt.scatter(x, y)
plt.savefig("lensing_posterior.png")
