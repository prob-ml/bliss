# %% [markdown]
# ## Imports

import gc

# %%
from os import environ
from pathlib import Path

import numpy as np
import torch
from astropy.visualization import make_lupton_rgb
from hydra import compose, initialize
from hydra.utils import instantiate
from matplotlib import pyplot as plt
from matplotlib.markers import MarkerStyle
from omegaconf import OmegaConf

from bliss.catalog import convert_mag_to_nmgy
from bliss.encoder.metrics import CatalogMatcher
from bliss.encoder.sample_image_renders import plot_plocs
from bliss.surveys.des import TractorFullCatalog
from bliss.surveys.sdss import PhotoFullCatalog

environ["CUDA_VISIBLE_DEVICES"] = "1"

torch.set_grad_enabled(False)

ckpt = "/home/regier/bliss/tests/data/base_config_trained_encoder.pt"
with initialize(config_path=".", version_base=None):
    cfg0 = compose(
        "config",
        {
            f"train.pretrained_weights={ckpt}",
            f"predict.weight_save_path={ckpt}",
            "cached_simulator.splits=0:80/80:90/99:100",
            "cached_simulator.num_workers=0",
        },
    )

cfg_c4 = OmegaConf.merge(cfg0, {"encoder": {"use_checkerboard": True, "n_sampler_colors": 4}})
cfg_c2 = OmegaConf.merge(
    cfg0,
    {
        "encoder": {
            "use_checkerboard": True,
            "n_sampler_colors": 2,
        }
    },
)
cfg_c1 = OmegaConf.merge(
    cfg0,
    {
        "encoder": {
            "use_checkerboard": False,
        }
    },
)

# %% [markdown]
# ## Load and view the SDSS field

# %%
sdss = instantiate(cfg0.surveys.sdss, load_image_data=True)
sdss.prepare_data()
(sdss_frame,) = sdss.predict_dataloader()  # noqa: WPS460
obs_image = sdss_frame["images"][0]

# %%
rgb = make_lupton_rgb(obs_image[3], obs_image[2], obs_image[1], Q=0, stretch=0.1)
plt.imshow(rgb, origin="lower")

# %% [markdown]
# ## Load and view SDSS predictions

# %%
rcf = cfg0.surveys.sdss.fields[0]

run, camcol, field = rcf["run"], rcf["camcol"], rcf["fields"][0]
po_fn = f"photoObj-{run:06d}-{camcol}-{field:04d}.fits"
po_path = Path(cfg0.paths.sdss) / str(run) / str(camcol) / str(field) / po_fn

sdss_wcs = sdss[0]["wcs"][2]
photo_cat = PhotoFullCatalog.from_file(po_path, sdss_wcs, *obs_image[2].shape)

# %% [markdown]
# ## Load and view DECaLS predictions

# %%
sdss_wcs = sdss[0]["wcs"][2]
decals_path = Path(cfg0.paths.des) / "336" / "3366m010" / "tractor-3366m010.fits"
decals_cat = TractorFullCatalog.from_file(decals_path, sdss_wcs, 1488, 2048)

# %% [markdown]
# ## Make and plot predictions with BLISS

# %%
gc.collect()
torch.cuda.empty_cache()

# %%
encoder = instantiate(cfg0.train.encoder).cuda()
enc_state_dict = torch.load(cfg0.train.pretrained_weights)
if cfg0.train.pretrained_weights.endswith(".ckpt"):
    enc_state_dict = enc_state_dict["state_dict"]
encoder.load_state_dict(enc_state_dict)
encoder.eval()

batch = {
    "images": obs_image[:, :, :].unsqueeze(0).cuda(),
    "psf_params": sdss_frame["psf_params"].cuda(),
}

# %%
bliss_tile_cat = encoder.sample(batch, use_mode=True)
bliss_tile_cat["n_sources"][0][0].fill_(0)
bliss_flux_filter_cat = bliss_tile_cat.filter_by_flux(convert_mag_to_nmgy(22.5))
bliss_cat = bliss_flux_filter_cat.to_full_catalog(4).to("cpu")

# %% [markdown]
# ## Three-way performance scoring

# %%
photo_cat_box = photo_cat.filter_by_ploc_box(torch.zeros(2), 1488)
decals_cat_box = decals_cat.filter_by_ploc_box(torch.zeros(2), 1488)
bliss_cat_box = bliss_cat.filter_by_ploc_box(torch.zeros(2), 1488)

# %%
# Create a CatalogMatcher object
matcher = CatalogMatcher()

# Match the catalogs based on their positions
match_gt_bliss = matcher.match_catalogs(decals_cat_box, bliss_cat_box)[0]
match_gt_photo = matcher.match_catalogs(decals_cat_box, photo_cat_box)[0]

# %%
fig, ax = plt.subplots(figsize=(16, 16))
bw = np.array(rgb, dtype=np.float32).sum(2)
ax.imshow(bw[:1488, :1488], origin="lower", cmap="gray")

plot_plocs(
    photo_cat_box,
    ax,
    0,
    "all",
    color="g",
    marker="X",
    s=50,
    edgecolor="black",
    linewidth=0.5,
    label="SDSS",
)
plot_plocs(
    bliss_cat_box,
    ax,
    0,
    "all",
    color="y",
    marker="P",
    s=30,
    edgecolor="black",
    linewidth=0.5,
    label="BLISS",
)

matches = {
    # in gt and pred or comp
    "gt_all": set(match_gt_bliss[0].numpy()).union(match_gt_photo[0].numpy()),
    # in pred and gt, not in comp
    "gt_pred_only": set(match_gt_bliss[0].numpy()).difference(match_gt_photo[0].numpy()),
    # in comp and gt, not in pred
    "gt_comp_only": set(match_gt_photo[0].numpy()).difference(match_gt_bliss[0].numpy()),
    # in pred, not in gt
    "pred_only": set(range(bliss_cat_box["n_sources"].item())).difference(
        match_gt_bliss[1].numpy()
    ),
    # in comp, not in gt
    "comp_only": set(range(photo_cat_box["n_sources"].item())).difference(
        match_gt_photo[1].numpy()
    ),
}

params = {
    "marker": MarkerStyle("o", fillstyle="none"),
    "s": 200,
    "linewidth": 0.7,
}
colors = [
    "#08F7FE",  # cyan
    "#FE53BB",  # pink
    "#F5D300",  # yellow
    "#00ff41",  # matrix green
]

plot_plocs(
    decals_cat_box,
    ax,
    0,
    list(matches["gt_pred_only"]),
    c=colors[0],
    label=r"(BLISS $\cup$ DECaLS) - SDSS",
    **params,
)
plot_plocs(
    decals_cat_box,
    ax,
    0,
    list(matches["gt_comp_only"]),
    c=colors[1],
    label=r"(SDSS $\cup$ DECaLS) - BLISS",
    **params,
)
plot_plocs(
    bliss_cat_box,
    ax,
    0,
    list(matches["pred_only"]),
    c=colors[2],
    label="BLISS - DECaLS",
    **params,
)
plot_plocs(
    photo_cat_box,
    ax,
    0,
    list(matches["comp_only"]),
    c=colors[3],
    label="SDSS - DECaLS",
    **params,
)

handles, labels = ax.get_legend_handles_labels()
ax.legend(
    handles=handles,
    labels=labels,
    loc="upper center",
    ncol=7,
    bbox_to_anchor=(0.0, 0.06, 1, 1),
    fontsize=10,
)
# %%
