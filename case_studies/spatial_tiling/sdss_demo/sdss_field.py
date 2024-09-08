# %% [markdown]
# ## Imports

import gc

# %%
from os import environ
from pathlib import Path

import numpy as np
import torch
from astropy.visualization import make_lupton_rgb
from einops import rearrange, repeat
from hydra import compose, initialize
from hydra.utils import instantiate
from matplotlib import pyplot as plt
from matplotlib.markers import MarkerStyle
from omegaconf import OmegaConf

from bliss.catalog import FullCatalog, TileCatalog, convert_mag_to_nmgy
from bliss.encoder.metrics import CatalogMatcher
from bliss.encoder.sample_image_renders import plot_plocs
from bliss.surveys.des import TractorFullCatalog
from bliss.surveys.sdss import PhotoFullCatalog

environ["CUDA_VISIBLE_DEVICES"] = "7"

torch.set_grad_enabled(False)

ckpt = "/home/regier/bliss_output/sep3_sdss_demo/version_3/checkpoints/best_encoder.ckpt"
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
photo_cat_base = PhotoFullCatalog.from_file(po_path, sdss_wcs, *obs_image[2].shape)

# filter fluxes
to_keep = photo_cat_base["fluxes"][:, :, 2] > convert_mag_to_nmgy(22.5)
d = {k: v[to_keep].unsqueeze(0) for k, v in photo_cat_base.items() if k != "n_sources"}
d["n_sources"] = to_keep.sum(1)
photo_cat = FullCatalog(photo_cat_base.height, photo_cat_base.width, d)

# %% [markdown]
# ## Load and view DECaLS predictions

# %%
sdss_wcs = sdss[0]["wcs"][2]
decals_path = Path(cfg0.paths.des) / "336" / "3366m010" / "tractor-3366m010.fits"
decals_cat_base = TractorFullCatalog.from_file(decals_path, sdss_wcs, 1488, 2048)

# a bit less than 22.5 magnitude, our target
to_keep = decals_cat_base["fluxes"][..., 0] > 1
d = {k: v[to_keep].unsqueeze(0) for k, v in decals_cat_base.items() if k != "n_sources"}
d["n_sources"] = to_keep.sum(1)
decals_cat = FullCatalog(decals_cat_base.height, decals_cat_base.width, d)

# %% [markdown]
# ## Make predictions with BLISS

# %%
gc.collect()
torch.cuda.empty_cache()

# %%
# change the cfg here to try different checkerboard schemes
encoder = instantiate(cfg_c1.train.encoder).cuda()
enc_state_dict = torch.load(cfg0.train.pretrained_weights)
if cfg0.train.pretrained_weights.endswith(".ckpt"):
    enc_state_dict = enc_state_dict["state_dict"]
encoder.load_state_dict(enc_state_dict)
encoder.eval()

patches = obs_image[2:3, :1488, :1488].unfold(1, 80, 72).unfold(2, 80, 72)
patches_batch = rearrange(patches, "bands ht wt hp wp -> (ht wt) bands hp wp")
psf_batch = repeat(
    sdss_frame["psf_params"][:, 2:3], "b h w -> (repeat b) h w", repeat=patches_batch.size(0)
)

batch = {
    "images": patches_batch.cuda(),
    "psf_params": psf_batch.cuda(),
}

bliss_patch_cat = encoder.sample(batch, use_mode=True)

# %%

bliss_disjoint_cat = bliss_patch_cat.symmetric_crop(1)

d = {}
d["n_sources"] = rearrange(
    bliss_disjoint_cat["n_sources"],
    "(hp wp) ht wt -> 1 (hp ht) (wp wt)",
    hp=patches.shape[1],
    wp=patches.shape[2],
)
for k, v in bliss_disjoint_cat.items():
    if k != "n_sources":
        pattern = "(hp wp) ht wt d1 d2 -> 1 (hp ht) (wp wt) d1 d2"
        d[k] = rearrange(v, pattern, hp=patches.shape[1], wp=patches.shape[2])
bliss_tile_cat = TileCatalog(d)

bliss_flux_filter_cat = bliss_tile_cat.filter_by_flux(convert_mag_to_nmgy(22.7), band=0)
bliss_cat = bliss_flux_filter_cat.to_full_catalog(4).to("cpu")

# %% [markdown]
# ## Three-way performance scoring and plotting

# %%
corner = torch.ones(2) * 4
photo_cat_box = photo_cat.filter_by_ploc_box(corner, 1439)
decals_cat_box = decals_cat.filter_by_ploc_box(corner, 1439)
bliss_cat_box = bliss_cat.filter_by_ploc_box(torch.zeros(2), 1439)

corner = torch.tensor([250, 1200])
photo_cat_box = photo_cat_box.filter_by_ploc_box(corner, 200, exclude_box=True)
decals_cat_box = decals_cat_box.filter_by_ploc_box(corner, 200, exclude_box=True)
bliss_cat_box = bliss_cat_box.filter_by_ploc_box(corner, 200, exclude_box=True)

corner = torch.tensor([400, 1100])
photo_cat_box = photo_cat_box.filter_by_ploc_box(corner, 100, exclude_box=True)
decals_cat_box = decals_cat_box.filter_by_ploc_box(corner, 100, exclude_box=True)
bliss_cat_box = bliss_cat_box.filter_by_ploc_box(corner, 100, exclude_box=True)


# Create a CatalogMatcher object
matcher = CatalogMatcher(
    dist_slack=5.0,
    mag_band=2,
)

# Match the catalogs based on their positions
match_gt_bliss = matcher.match_catalogs(decals_cat_box, bliss_cat_box)[0]
match_gt_photo = matcher.match_catalogs(decals_cat_box, photo_cat_box)[0]

fig, ax = plt.subplots(figsize=(14, 14))
bw = np.array(rgb, dtype=np.float32).sum(2)
ax.imshow(bw[4:1444, 4:1444], origin="lower", cmap="gray")

plot_plocs(
    decals_cat_box,
    ax,
    0,
    "all",
    color="r",
    marker=MarkerStyle("o", fillstyle="none"),
    s=100,
    linewidth=0.5,
    label="DECaLS",
)
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
    # in decals and (bliss or sdss)
    "gt_all": set(match_gt_bliss[0].numpy()).union(match_gt_photo[0].numpy()),
    # in bliss and decals, not in sdss
    "bliss_tp_only": set(match_gt_bliss[0].numpy()).difference(match_gt_photo[0].numpy()),
    # in sdss and decals, not in bliss
    "sdss_tp_only": set(match_gt_photo[0].numpy()).difference(match_gt_bliss[0].numpy()),
    # in bliss, not in decals
    "bliss_fp": set(range(bliss_cat_box["n_sources"].item())).difference(match_gt_bliss[1].numpy()),
    # in sdss, not in decals
    "sdss_fp": set(range(photo_cat_box["n_sources"].item())).difference(match_gt_photo[1].numpy()),
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
    list(matches["bliss_tp_only"]),
    c=colors[0],
    label=r"(BLISS $\cap$ DECaLS) - SDSS",
    **params,
)
plot_plocs(
    decals_cat_box,
    ax,
    0,
    list(matches["sdss_tp_only"]),
    c=colors[1],
    label=r"(SDSS $\cap$ DECaLS) - BLISS",
    **params,
)
plot_plocs(
    bliss_cat_box,
    ax,
    0,
    list(matches["bliss_fp"]),
    c=colors[2],
    label="BLISS - DECaLS",
    **params,
)
plot_plocs(
    photo_cat_box,
    ax,
    0,
    list(matches["sdss_fp"]),
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

for k, v in matches.items():
    print(k, len(v))  # noqa: WPS421
# %%
