# %%
from pytorch_lightning import Trainer
from torch.utils.data.dataset import Dataset
from bliss.models.fnp import StarPCA
import numpy as np
import torch

from pathlib import Path

import bliss
from bliss.datasets import sdss
from bliss.plotting import plot_image, plot_image_locs, _plot_locs


import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

STAMPSIZE = 11

#%%
class SDSS(Dataset):
    def __init__(self, sdss_source, band=2, min_stars_in_field=100):
        super().__init__()
        self.band = band
        self.sdss_source = [s for s in sdss_source if len(s["prs"]) > min_stars_in_field]
        self.cached_items = [None] * len(self.sdss_source)

    def __len__(self):
        return len(self.cached_items)

    def __getitem__(self, idx):
        if self.cached_items[idx] is None:
            out = self.makeitem(idx)
            self.cached_items[idx] = out
        else:
            out = self.cached_items[idx]
        return out

    def makeitem(self, idx):
        data = self.sdss_source[idx]
        img = data["image"][self.band]
        locs = torch.stack((torch.from_numpy(data["prs"]), torch.from_numpy(data["pts"])), dim=1)
        X = (locs - locs.mean(0)) / locs.std(0)

        ## Randomize order
        idxs = np.random.choice(X.size(0), X.size(0), replace=False)
        X = X[idxs]
        locs = locs[idxs]

        return (X, img, locs)

    @staticmethod
    def make_G_from_clust(c, nclust=None):
        if not nclust:
            nclust = len(np.unique(c))
        G = torch.zeros((len(c), nclust))
        for i in range(nclust):
            G[c == i, i] = 1.0
        return G

    def plot_clustered_locs(self, idx):
        pl, axes = plt.subplots(nrows=1, ncols=1, figsize=(4, 4))
        plot_image(pl, axes, self.sdss_source[idx]["image"][2])
        _, _, locs = self[idx]
        km = KMeans(n_clusters=5)
        clst = km.fit_predict(locs.numpy())
        colors = ["red", "green", "blue", "orange", "yellow"]
        for i, cl in enumerate(np.unique(clst)):
            _plot_locs(axes, 1, 0, locs[clst == cl], color=colors[i], s=3)

        plot_image_locs(axes, 1, 0, km.cluster_centers_, colors=("white",))
        return pl


#%%
sdss_source = sdss.SloanDigitalSkySurvey(
    # sdss_dir=sdss_dir,
    Path(bliss.__file__).parents[1].joinpath("data/sdss_all"),
    run=3900,
    camcol=6,
    fields=range(300, 1000),
    # fields=(808,),
    bands=range(5),
)


#%%
# Pickle
sdss_dataset_file = Path("sdss_source.pkl")
#%%
if not sdss_dataset_file.exists():
    sdss_dataset = SDSS(sdss_source)
    torch.save(sdss_dataset, sdss_dataset_file)
else:
    sdss_dataset = torch.load(sdss_dataset_file)

#%%

#%%
m = StarPCA(k=4, n_clusters=5, stampsize=STAMPSIZE)

#%%
m.fit_dataset(sdss_dataset)
X, S, Y = m.prepare_batch(sdss_dataset[-1])
#%%
x = m.predict(X, Y)
#%%
import pandas as pd


def calc_mse(sdss_dataset, m):
    fields = []
    mses = []
    for (i, data) in enumerate(sdss_dataset):
        field = sdss_dataset.sdss_source[i]["field"]
        X, S, Y = m.prepare_batch(data)
        n_s = S.size(0)
        Y_hat = m.predict(X, S)
        mse = (Y - Y_hat).pow(2).mean()
        fields.append(field)
        mses.append(mse.item())
    # fields, mses = list(zip(res))
    return pd.DataFrame({"field": fields, "mses": mses})


mses = calc_mse(sdss_dataset, m)
mses.to_csv("mses_pca.csv")
# %%
