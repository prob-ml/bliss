# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# # Report 2020-09-16
#
# ## Overview
#
# Here we compare the original Functional Neural Process (FNP) and the FNP+ to our newer attention-based models.
# Specifically, the four models considered are:
# -  FNP
# -  FNP+
# -  Attention FNP (ours)
# -  Deep-set FNP (ours)
#
# The Attention-based FNP replaces the averaging of reference points in the FNP with a set transformer architecture.
# These apply Self-Attention Blocks (SABs) to the set, then use a Pooling-by-Multihead-Attention (PMA) block to aggregate.
#
# The Deep-set FNP applies a neural-net to each member of the set, adds them together, then applies another neural-net.
#
# Both the Set Transformer and Deep Set architectures are universal approximators of set-valued functions. However, we would expect the Set Transformer to perform better empirically with the attention mechanism explicity introducing interactions between elements of the input set.
#
# %% [markdown]
# ## Synthetic DGP
# We use a synthetic data generating process which is identical to the one used in the FNP paper, except that there are now 100 indepedent samples. Each sample is a translation of the input by a random amount; i.e.
#
# $$
# f_i(X) = f(X + \epsilon_i)~\epsilon_i \sim N(0, 1)
# $$
#
# # Results
# We show very favorable performance of the attention-based model compared to the original FNP and FNP+. The FNP and FNP+ learn a very noisy prediction. Meanwhile, the attention-based model can successfully utilize information across the realizations of the function to properly interpolate.
#
# The deep set FNP performs fairly well, but not as well as the attention-based model.
#
# # Next steps
# -  Comparison/acknowledgement of methods like the Attention Neural Process. Our method is different since it builds on the sparse dependency graph and exchangeable structure of the FNP.

# %%
import os
import sys

path = os.path.abspath("..")
if path not in sys.path:
    sys.path.insert(0, path)
import numpy as np
from bliss.models.fnp import (
    FNP,
    SequentialVarg,
    MLP,
    SplitLayer,
    NormalEncoder,
    AveragePooler,
    SetPooler,
    RepEncoder,
    ConcatLayer,
    DepGraph,
)
from sklearn.preprocessing import StandardScaler
import torch
from torch.optim import Adam
from scipy.signal import savgol_filter
import warnings
import math
import matplotlib.pyplot as plt

plt.style.use(["seaborn-whitegrid", "seaborn-colorblind", "seaborn-notebook"])
VISUALIZE = True
warnings.filterwarnings("ignore")
# %%
def train_onedim_model(model, od, epochs=10000, lr=1e-4, visualize=False):
    if torch.cuda.is_available():
        model = model.cuda()
    optimizer = Adam(model.parameters(), lr=lr)
    model.train()
    holdout_loss_prev = np.infty
    holdout_loss_initial = model(od.XR, od.yhR, od.XM, od.yhM)
    holdout_loss_best = holdout_loss_initial
    print("Initial holdout loss: {:.3f})".format(holdout_loss_initial.item()))
    stdy = od.stdy
    for i in range(epochs):
        optimizer.zero_grad()

        loss = model(od.XR, od.yR, od.XM, od.yM)
        loss.backward()
        optimizer.step()

        if i % int(epochs / 10) == 0:
            print("Epoch {}/{}, loss: {:.3f}".format(i, epochs, loss.item()))
            if visualize:
                visualize_onedim(
                    model,
                    od.dx,
                    od.stdx,
                    None,
                    od.f,
                    cond_x=od.XR,
                    cond_y=od.yR[0],
                    all_x=od.X,
                    all_y=od.y[0],
                    range_y=(-2.0, 3.0),
                    samples=100,
                )
            holdout_loss = model(od.XR, od.yhR, od.XM, od.yhM)
            if holdout_loss < holdout_loss_best:
                holdout_loss_best = holdout_loss
            print("Holdout loss: {:.3f}".format(holdout_loss.item()))
    if visualize:
        visualize_onedim(
            model,
            od.dx,
            od.stdx,
            stdy,
            od.f,
            cond_x=od.XR,
            cond_y=od.yR[0],
            all_x=od.X,
            all_y=od.y[0],
            range_y=(-2.0, 3.0),
            samples=100,
        )
    print("Done.")
    return model, holdout_loss_initial, holdout_loss, holdout_loss_best


# %%
def visualize_onedim(
    model,
    dx,
    stdx,
    stdy,
    f,
    cond_x=None,
    cond_y=None,
    all_x=None,
    all_y=None,
    samples=30,
    range_y=(-100.0, 100.0),
    title="",
    train=False,
):
    """
    Visualizes the predictive distribution
    """
    dxy = np.zeros((dx.shape[0], samples))
    if not train:
        model.eval()
    with torch.no_grad():
        dxi = torch.from_numpy(stdx.transform(dx).astype(np.float32))
        if torch.cuda.is_available():
            dxi = dxi.cuda()
        for j in range(samples):
            dxy[:, j] = (
                model.predict(dxi, cond_x, cond_y.unsqueeze(0)).cpu().numpy().ravel()
            )
    print()

    plt.figure()
    mean_dxy, std_dxy = dxy.mean(axis=1), dxy.std(axis=1)
    # smooth it in order to avoid the sampling jitter
    mean_dxys = savgol_filter(mean_dxy, 61, 3)
    std_dxys = savgol_filter(std_dxy, 61, 3)

    if torch.cuda.is_available():
        all_x, all_y, cond_x, cond_y = (
            all_x.cpu(),
            all_y.cpu(),
            cond_x.cpu(),
            cond_y.cpu(),
        )

    if stdy is None:
        y_invt = lambda x: x
    else:
        y_invt = stdy.inverse_transform

    plt.plot(dx.ravel(), mean_dxys, label="Mean function")
    dxf = f(dx, 0.0)
    plt.plot(dx.ravel(), dxf.ravel(), label="Actual function")
    plt.plot(
        stdx.inverse_transform(all_x.data.numpy()).ravel(),
        y_invt(all_y.data.numpy()).ravel(),
        "o",
        label="Observations",
    )
    if cond_x is not None:
        plt.plot(
            stdx.inverse_transform(cond_x.data.numpy()).ravel(),
            y_invt(cond_y.data.numpy()).ravel(),
            "o",
            label="Reference",
        )
    plt.fill_between(
        dx.ravel(), mean_dxys - 1.0 * std_dxys, mean_dxys + 1.0 * std_dxys, alpha=0.1
    )
    plt.fill_between(
        dx.ravel(), mean_dxys - 2.0 * std_dxys, mean_dxys + 2.0 * std_dxys, alpha=0.1
    )
    plt.fill_between(
        dx.ravel(), mean_dxys - 3.0 * std_dxys, mean_dxys + 3.0 * std_dxys, alpha=0.1
    )

    plt.xlim([np.min(dx), np.max(dx)])
    plt.ylim(range_y)
    plt.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.10),
        ncol=3,
        fancybox=False,
        shadow=False,
    )
    plt.title(title)
    model.train()
    plt.show()


# %%
class OneDimDataset:
    def __init__(
        self,
        N=20,
        num_extra=500,
        seed=1,
        offset=0.1,
    ):
        ## Generate the first row as in the FNP paper
        np.random.seed(seed)
        X = np.concatenate(
            [
                np.random.uniform(low=0, high=0.6, size=(N - 8, 1)),
                np.random.uniform(low=0.8, high=1.0, size=(8, 1)),
            ],
            axis=0,
        )
        eps = np.random.normal(0.0, 0.03, size=(X.shape[0], 1))
        self.f = lambda x, eps: x + np.sin(4 * (x + eps)) + np.sin(13 * (x + eps)) + eps
        y = self.f(X, eps)

        ## Pick which indices are references or not
        # self.idxR = idxR
        # self.idxM = np.array([i for i in idx if i not in idxR.tolist()])

        ## Generate more y-values
        ys = [y]
        for _ in range(99):
            Xi = X + np.random.normal()
            eps_i = np.random.normal(0.0, 0.03, size=(X.shape[0], 1))
            # f = lambda x, eps: x + np.sin(4 * (x + eps)) + np.sin(13 * (x + eps)) + eps
            yi = self.f(Xi, eps_i)
            ys.append(yi)
        y = np.concatenate(ys, axis=1).transpose()

        ## Generate holdouts
        ys = []
        for i in range(10):
            Xi = X + np.random.normal()
            eps_i = np.random.normal(0.0, 0.03, size=(X.shape[0], 1))
            # f = lambda x, eps: x + np.sin(4 * (x + eps)) + np.sin(13 * (x + eps)) + eps
            yi = self.f(Xi, eps_i)
            ys.append(yi)
        yh = np.concatenate(ys, axis=1).transpose()

        self.stdx, self.stdy = StandardScaler().fit(X), StandardScaler().fit(
            y.reshape(-1, 1)
        )
        # X, y = stdx.transform(X), stdy.transform(y)
        X = self.stdx.transform(X)
        idx = np.arange(X.shape[0])
        # self.idxR = np.random.choice(idx, size=(10,), replace=False)
        # self.idxM = np.array([i for i in idx if i not in idxR.tolist()])
        self.idxR = np.array([2, 16, 9, 6, 17, 12, 4, 15, 1, 14])
        self.idxM = np.array([i for i in idx if i not in self.idxR.tolist()])

        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.float32)).unsqueeze(2)
        self.XR, self.yR = self.X[self.idxR], self.y[:, self.idxR]
        self.XM, self.yM = self.X[self.idxM], self.y[:, self.idxM]

        ## Holdouts
        yh = torch.from_numpy(yh.astype(np.float32))
        self.yh = yh.unsqueeze(2)
        self.yhR = self.yh[:, self.idxR]
        self.yhM = self.yh[:, self.idxM]

        ## Point where predictions will be made for plotting
        self.dx = np.linspace(-1.0, 2.0, num_extra).astype(np.float32)[:, np.newaxis]

    def cuda(self):
        for nm in ["XR", "XM", "X", "yR", "yM", "y", "yhR", "yhM", "yh"]:
            setattr(self, nm, getattr(self, nm).cuda())

    def cpu(self):
        for nm in ["XR", "XM", "X", "yR", "yM", "y", "yhR", "yhM", "yh"]:
            setattr(self, nm, getattr(self, nm).cpu())

        # return x, y, dx, f


# %%
od = OneDimDataset()
# %%
if torch.cuda.is_available():
    od.cuda()

# %%
def make_onedim_model(
    dim_x=1,
    dim_y=1,
    dim_z=50,
    dim_u=3,
    dim_y_enc=100,
    fb_z=1.0,
    use_plus=False,
    use_set_pooler=False,
    pooling_rep_size=32,
    pooling_layers=[64],
    use_attention=False,
    st_numheads=[2, 2],
):
    cov_vencoder = SequentialVarg(
        MLP(dim_x, [100], 2 * dim_u),
        SplitLayer(dim_u, -1),
        NormalEncoder(),
    )
    dep_graph = DepGraph(dim_u)
    trans_cond_y = MLP(dim_y, [128], dim_y_enc)
    if use_set_pooler:
        rep_encoder = RepEncoder(
            MLP(dim_y_enc + dim_u, [128], pooling_rep_size),
            use_u_diff=True,
            use_x=False,
        )
        pooler = SequentialVarg(
            SetPooler(
                pooling_rep_size,
                dim_z,
                pooling_layers,
                use_attention,
                st_numheads,
            ),
            SplitLayer(dim_z, -1),
            NormalEncoder(minscale=1e-8),
        )
    else:
        rep_encoder = RepEncoder(
            MLP(dim_y_enc + dim_x, [128], 2 * dim_z), use_u_diff=False, use_x=True
        )
        pooler = SequentialVarg(
            AveragePooler(dim_z),
            SplitLayer(dim_z, -1),
            NormalEncoder(minscale=1e-8),
        )
    prop_vencoder = SequentialVarg(
        ConcatLayer([1, 0]),
        MLP(
            dim_y_enc + dim_x,
            [128],
            2 * dim_z,
        ),
        SplitLayer(dim_z, -1),
        NormalEncoder(minscale=1e-8),
    )
    output_in = [0] if not use_plus else [0, 1]
    output_insize = dim_z if not use_plus else dim_z + dim_u
    label_vdecoder = SequentialVarg(
        ConcatLayer(output_in),
        MLP(output_insize, [128], 2 * dim_y),
        SplitLayer(dim_y, -1),
        NormalEncoder(minscale=0.1),
    )
    model = FNP(
        cov_vencoder,
        dep_graph,
        trans_cond_y,
        rep_encoder,
        pooler,
        prop_vencoder,
        label_vdecoder,
        fb_z=fb_z,
    )

    return model


# %% [markdown]
# ## FNP

# %%
torch.manual_seed(5)


# %%
# fnp = RegressionFNP(
#     dim_x=1,
#     dim_y=1,
#     transf_y=od.stdy,
#     dim_h=100,
#     dim_u=3,
#     n_layers=1,
#     dim_z=50,
#     fb_z=1.0,
#     use_plus=False,
# )
dim_x = 1
dim_y = 1
dim_z = 50
dim_u = 3
dim_y_enc = 100
fnp = make_onedim_model()
if torch.cuda.is_available():
    fnp = fnp.cuda()

fnp, _, _, fnp_loss_best = train_onedim_model(
    fnp, od, epochs=10000, lr=1e-4, visualize=VISUALIZE
)

# %% [markdown]
# ## FNP+

# %%
fnpp = make_onedim_model(use_plus=True)
fnpp, _, fnpp_loss_final = train_onedim_model(
    fnpp, od, epochs=10000, lr=1e-4, visualize=VISUALIZE
)

# %% [markdown]
# ## Attention FNP

# %%
attt = make_onedim_model(use_set_pooler=True, use_attention=True)
attt, _, attt_loss_final = train_onedim_model(
    attt, od, epochs=10000, lr=1e-4, visualize=VISUALIZE
)

# %% [markdown]
# ## Deep-set FNP

# %%
poolnp = make_onedim_model(use_set_pooler=False, use_attention=False, fb_z=0.5)
poolnp, _, poolnp_loss_final = train_onedim_model(
    poolnp, od, epochs=10000, lr=1e-4, visualize=VISUALIZE
)
