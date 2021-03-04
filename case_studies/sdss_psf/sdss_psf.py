# %%
import numpy as np
import torch

from pathlib import Path
from astropy.io import fits
from astropy.wcs import WCS

import bliss
from bliss.datasets import sdss
from bliss.plotting import plot_image, plot_image_locs, _plot_locs

import matplotlib.pyplot as plt

sdss_data = sdss.SloanDigitalSkySurvey(
    # sdss_dir=sdss_dir,
    Path(bliss.__file__).parents[1].joinpath("data/sdss"),
    run=3900,
    camcol=6,
    fields=(269,),
    bands=range(5),
)

image = torch.Tensor(sdss_data[0]["image"])
slen0 = image.shape[-2]
slen1 = image.shape[-1]

plt.imshow(image[2, :100, :100])

#%%
fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(4,4))
plot_image(fig, axes, image[2])
locs = torch.stack((torch.from_numpy(sdss_data[0]["prs"]), torch.from_numpy(sdss_data[0]["pts"])), dim=1)
plot_image_locs(axes, slen=1, border_padding=0, true_locs=locs)

# %%
fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(4,4))
plot_image(fig, axes, image[2])
locs = torch.stack((torch.from_numpy(sdss_data[0]["prs"]), torch.from_numpy(sdss_data[0]["pts"])), dim=1)
#plot_image_locs(axes, slen=1, border_padding=0, true_locs=locs)
axes.set_xlim(1200, 1400)
axes.set_ylim(1050, 1200)
# %%
sdss_data[0]["bright_stars"]
# %%
from sklearn.cluster import KMeans
# %%
km=KMeans(n_clusters=5)
c=km.fit_predict(locs)
def plot_clustered_locs(axes, clst, locs):
    colors = ["red", "green", "blue", "orange", "yellow"]
    for i, cl in enumerate(np.unique(clst)):
        _plot_locs(axes, 1, 0, locs[clst==cl], color=colors[i], s=3)

# %%
c
# %%
# %%
fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(4,4))
plot_image(fig, axes, image[2])
plot_clustered_locs(axes, c, locs)
plot_image_locs(axes, 1, 0, km.cluster_centers_, colors=("white", ))
# %%
