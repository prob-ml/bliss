import sys
import copy
import pickle

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

# plt.style.use("ggplot")
import torch

from astropy.table import Table
import plotly.express as px
import plotly.graph_objects as go
from hydra import compose, initialize
from hydra.utils import instantiate
import numpy as np

import seaborn as sns
sns.set_style("dark")
plt.rcParams['text.usetex'] = True
plt.rcParams["axes.grid"] = False
plt.rcParams["font.size"] = 24


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

test_type = "lensed_"
batch_size = 8
trials = 1000
min_bucket_size = 25

for _ in range(trials):

    tile_catalog = dataset.sample_prior(
        batch_size, cfg.datasets.simulated.n_tiles_h, cfg.datasets.simulated.n_tiles_w
    )
    tile_catalog.set_all_fluxes_and_mags(dataset.image_decoder)
    images, backgrounds = dataset.simulate_image_from_catalog(tile_catalog)

    tile_map = enc.variational_mode(images, backgrounds, tile_catalog)
    tile_map.locs = copy.deepcopy(tile_catalog.locs)
    tile_map.n_sources = copy.deepcopy(tile_catalog.n_sources)

    full_map = tile_map.cpu().to_full_params()
    full_true = tile_catalog.cpu().to_full_params()

    for i in range(batch_size):
        true_gal_bool = full_true[f"{test_type}galaxy_bools"].cpu().numpy()[i, :, 0].astype("bool").squeeze()
        pred_gal_probs = full_map[f"{test_type}galaxy_probs"].cpu().numpy()[i, ...].squeeze()
        
        # true_plocs = full_true.plocs.cpu().numpy().squeeze()[i,...]
        # pred_plocs = full_map.plocs.cpu().numpy().squeeze()[i,...]

        gal_probs = pred_gal_probs[true_gal_bool]
        gal_buckets = (gal_probs // delta).astype("int")
        for bucket in gal_buckets:
            galaxies[bucket] += 1

        count_buckets = (pred_gal_probs // delta).astype("int")
        for bucket in count_buckets:
            counts[bucket] += 1
    
x = []
y = []
yerr = []
for i, (galaxy, count) in enumerate(zip(galaxies, counts)):
    if count > min_bucket_size:
        x.append(increments[i])

        phat = galaxy / count
        y.append(phat)
        yerr.append(np.sqrt(phat * (1 - phat) / count))
data = x, y

with open(f"{test_type}galaxy_posterior_new.pkl", "wb") as f:
    pickle.dump(data, f)

with open(f"{test_type}galaxy_posterior_new.pkl", "rb") as f:
    data = pickle.load(f)
    x, y = data

if test_type == "lensed_":
    plt.xlabel("$\gamma_s$")
else:
    plt.xlabel("$a_s$")

plt.locator_params(axis='y', nbins=3)
plt.ylabel(r"$\mathrm{Empirical Coverage}$")
plt.plot(increments, increments, color="b")
plt.errorbar(x, y, fmt="o")
plt.savefig(f"test.png")

plt.savefig(f"{test_type}galaxy_posterior_new.png")