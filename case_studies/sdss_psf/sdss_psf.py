# %%
from pytorch_lightning import LightningModule, Trainer
from torch.utils.data.dataloader import DataLoader
from bliss.utils import ConcatLayer, MLP, NormalEncoder, SequentialVarg, SplitLayer
from bliss.models.fnp import AveragePooler, HNP
import numpy as np
import torch

from pathlib import Path
from astropy.io import fits
from astropy.wcs import WCS
from torch.optim import Adam
from torch.nn import Linear

import bliss
from bliss.datasets import sdss
from bliss.plotting import plot_image, plot_image_locs, _plot_locs
import bliss.models.fnp as fnp

from torch.distributions import Normal

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
fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(4, 4))
plot_image(fig, axes, image[2])
locs = torch.stack(
    (torch.from_numpy(sdss_data[0]["prs"]), torch.from_numpy(sdss_data[0]["pts"])), dim=1
)
plot_image_locs(axes, slen=1, border_padding=0, true_locs=locs)

# %%
fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(4, 4))
plot_image(fig, axes, image[2])
locs = torch.stack(
    (torch.from_numpy(sdss_data[0]["prs"]), torch.from_numpy(sdss_data[0]["pts"])), dim=1
)
# plot_image_locs(axes, slen=1, border_padding=0, true_locs=locs)
axes.set_xlim(1200, 1400)
axes.set_ylim(1050, 1200)
# %%
sdss_data[0]["bright_stars"]
# %%
from sklearn.cluster import KMeans

# %%
km = KMeans(n_clusters=5)
c = km.fit_predict(locs)


def plot_clustered_locs(axes, clst, locs):
    colors = ["red", "green", "blue", "orange", "yellow"]
    for i, cl in enumerate(np.unique(clst)):
        _plot_locs(axes, 1, 0, locs[clst == cl], color=colors[i], s=3)


# %%
c
# %%
# %%
fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(4, 4))
plot_image(fig, axes, image[2])
plot_clustered_locs(axes, c, locs)
plot_image_locs(axes, 1, 0, km.cluster_centers_, colors=("white",))
# %%
## Estimate this image
class SDSS_HNP(LightningModule):
    def __init__(self, dy=81, dz=4):
        dh = 2 * dz
        super().__init__()
        z_inference = SequentialVarg(
            ConcatLayer([1]), MLP(dy, [64, 32, 16], 2 * dz), SplitLayer(dz, -1), NormalEncoder()
        )

        z_prior = SequentialVarg(fnp.AveragePooler(dz), SplitLayer(dz, -1), NormalEncoder())

        h_prior = lambda X, G: Normal(
            torch.zeros(G.size(1), dh, device=G.device), torch.ones(G.size(1), dh, device=G.device)
        )

        h_pooler = SequentialVarg(
            fnp.AveragePooler(dh, f=Linear(2 * dz, 2 * dh)), SplitLayer(dh, -1), NormalEncoder()
        )

        y_decoder = SequentialVarg(
            ConcatLayer([0]),
            MLP(dz, [16, 32, 64, 128], 2 * dy),
            SplitLayer(dy, -1),
            NormalEncoder(),
        )
        self.hnp = fnp.HNP(z_inference, z_prior, h_prior, h_pooler, y_decoder)
        self.valid_losses = []

    def training_step(self, batch, batch_idx):
        X, G, S, Y = batch
        loss = self.hnp(X, G, S, Y)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.training_step(batch, batch_idx)
        self.log("val_loss", loss)
        self.valid_losses.append(loss.item())
        return loss

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=1e-3)


def make_G_from_clust(c, nclust=None):
    if not nclust:
        nclust = len(np.unique(c))
    G = torch.zeros((len(c), nclust))
    for i in range(nclust):
        G[c == i, i] = 1.0
    return G


# %%
## Prepare dataset
X = (locs - locs.mean(0)) / locs.std(0)
G = make_G_from_clust(c)
## Randomize order
idxs = np.random.choice(len(c), len(c), replace=False)
X = X[idxs]
G = G[idxs]
Y = torch.from_numpy(sdss_data[0]["bright_stars"])[idxs].reshape(-1, 25)
Y = (Y - Y.mean(1, keepdim=True)) / Y.std(1, keepdim=True)
S = Y[0:20]
# %%

trainloader = DataLoader([[X, G, S, Y]], batch_size=None, batch_sampler=None)

m = SDSS_HNP(25, 4)
# %%
trainer = Trainer(max_epochs=400, checkpoint_callback=False)
trainer.fit(m, trainloader)
# %%
x = m.hnp.predict(X, G, S)
# %%
plt.imshow(x[20].reshape(5, 5))
# %%
plt.imshow(Y[20].reshape(5, 5))
# %%
def plot_cluster_images(c, y_true, y_pred):
    n_max = 0
    for i in np.unique(c):
        if sum(c == i) > n_max:
            n_max = sum(c == i)
    figsize = (20, 20)
    plot, axes = plt.subplots(nrows=len(np.unique(c)), ncols=n_max * 2, figsize=figsize)
    for i in np.unique(c):
        ytc = y_true[c == i]
        ypc = y_pred[c == i]
        for j in range(ytc.shape[0]):
            ax = axes[i, 2 * j]
            ax.imshow(ytc[j].reshape(5, 5))
            ax = axes[i, 2 * j + 1]
            ax.imshow(ypc[j].reshape(5, 5))
    return plot, axes


p, a = plot_cluster_images(c, Y, x)
p.savefig("test.png")
# %%
