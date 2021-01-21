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
from bliss.models.fnp import RegressionFNP, PoolingFNP
from sklearn.preprocessing import StandardScaler
import torch
from torch.optim import Adam
from scipy.signal import savgol_filter
import warnings
import math
import matplotlib.pyplot as plt
plt.style.use(['seaborn-whitegrid', 'seaborn-colorblind', 'seaborn-notebook'])
warnings.filterwarnings('ignore')


# %%
def visualize(model, dx, stdx, stdy, f, cond_x=None, cond_y=None, all_x=None, all_y=None, samples=30, 
              range_y=(-100., 100.), title='', train=False):
    '''
    Visualizes the predictive distribution
    '''
    dxy = np.zeros((dx.shape[0], samples))
    if not train:
        model.eval()
    with torch.no_grad():
        dxi = torch.from_numpy(stdx.transform(dx).astype(np.float32))
        if torch.cuda.is_available():
            dxi = dxi.cuda()
        for j in range(samples):
            dxy[:, j] = model.predict(dxi, cond_x, cond_y.unsqueeze(0)).cpu().numpy().ravel()
    print()
 
    plt.figure()
    mean_dxy, std_dxy = dxy.mean(axis=1), dxy.std(axis=1)
    # smooth it in order to avoid the sampling jitter
    mean_dxys = savgol_filter(mean_dxy, 61, 3)
    std_dxys = savgol_filter(std_dxy, 61, 3)
    
    if torch.cuda.is_available():
        all_x, all_y, cond_x, cond_y = all_x.cpu(), all_y.cpu(), cond_x.cpu(), cond_y.cpu()

    if stdy is None:
        y_invt = lambda x: x
    else:
        y_invt = stdy.inverse_transform

    plt.plot(dx.ravel(), mean_dxys, label='Mean function')
    dxf = f(dx, 0.0)
    plt.plot(dx.ravel(), dxf.ravel(), label = 'Actual function')
    plt.plot(stdx.inverse_transform(all_x.data.numpy()).ravel(), y_invt(all_y.data.numpy()).ravel(), 'o',
             label='Observations')
    if cond_x is not None:
        plt.plot(stdx.inverse_transform(cond_x.data.numpy()).ravel(), y_invt(cond_y.data.numpy()).ravel(), 'o',
             label='Reference')
    plt.fill_between(dx.ravel(), mean_dxys-1.*std_dxys, mean_dxys+1.*std_dxys, alpha=.1)
    plt.fill_between(dx.ravel(), mean_dxys-2.*std_dxys, mean_dxys+2.*std_dxys, alpha=.1)
    plt.fill_between(dx.ravel(), mean_dxys-3.*std_dxys, mean_dxys+3.*std_dxys, alpha=.1)

    plt.xlim([np.min(dx), np.max(dx)])
    plt.ylim(range_y)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.10), ncol=3, fancybox=False, shadow=False)
    plt.title(title)
    model.train()
    plt.show()


# %%
def dataset():
    N, num_extra = 20, 500
    N_rows = 100
    np.random.seed(1)
    x = np.concatenate([np.random.uniform(low=0, high=0.6, size=(N-8, 1)),
                        np.random.uniform(low=0.8, high=1., size=(8, 1))], axis=0)
    offset = 0.1
    eps = np.random.normal(0., 0.03, size=(x.shape[0], 1))
    f = lambda x, eps: x + np.sin(4 * (x + eps)) + np.sin(13 * (x + eps)) + eps
    y = f(x, eps)
    
    dx = np.linspace(-1., 2., num_extra).astype(np.float32)[:, np.newaxis]
    return x, y, dx, f


# %%
def dataset_offset(x):
    N, num_extra = 20, 500
    N_rows = 100
    x = x + np.random.normal()
    offset = 0.1
    eps = np.random.normal(0., 0.03, size=(x.shape[0], 1))
    f = lambda x, eps: x + np.sin(4 * (x + eps)) + np.sin(13 * (x + eps)) + eps
    y = f(x, eps)
    return y


# %%
def train_model(model, epochs=10000, lr=1e-4):
    if torch.cuda.is_available():
        model = model.cuda()
    optimizer = Adam(model.parameters(), lr=lr)
    model.train()
    holdout_loss_prev = np.infty
    for i in range(epochs):
        optimizer.zero_grad()

        loss = model(XR, yR, XM, yM)
        loss.backward()
        optimizer.step()

        if i % int(epochs / 10) == 0:
            print('Epoch {}/{}, loss: {:.3f}'.format(i, epochs, loss.item()))

            visualize(model, dx, stdx, None, f, cond_x=XR, cond_y=yR[0], all_x=X, all_y=y[0], range_y=(-2., 3.), samples=100)
            holdout_loss = model(XR, yhR, XM, yhM)[0]
            print('Holdout loss: {:.3f}'.format(holdout_loss))
    visualize(model, dx, stdx, None, f, cond_x=XR, cond_y=yR[0], all_x=X, all_y=y[0], range_y=(-2., 3.), samples=100)
    print('Done.')
    return model


# %%
X, yb, dx, f = dataset()
idx = np.arange(X.shape[0])
#idxR = np.random.choice(idx, size=(10,), replace=False)
idxR = np.array([ 2,  4,  6, 12, 16, 13,  3,  5,  9,  1])
idxM = np.array([i for i in idx if i not in idxR.tolist()])
ys = [yb]
for i in range(99):
    ys.append(dataset_offset(X))
y=np.concatenate(ys, axis=1).transpose()
ys = []
for i in range(10):
    ys.append(dataset_offset(X))
yh=np.concatenate(ys, axis=1).transpose()

stdx, stdy = StandardScaler().fit(X), StandardScaler().fit(y.reshape(-1, 1))
#X, y = stdx.transform(X), stdy.transform(y)
X = stdx.transform(X)
idx = np.arange(X.shape[0])
idxR = np.random.choice(idx, size=(10,), replace=False)
idxM = np.array([i for i in idx if i not in idxR.tolist()])

X, y = torch.from_numpy(X.astype(np.float32)), torch.from_numpy(y.astype(np.float32))
y = y.unsqueeze(2)
XR, yR = X[idxR], y[:, idxR]
XM, yM = X[idxM], y[:, idxM]


# %%
yh = torch.from_numpy(yh.astype(np.float32))
yh = yh.unsqueeze(2)
yhR = yh[:, idxR]
yhM = yh[:, idxM]


# %%
if torch.cuda.is_available():
    XR, XM, X = XR.cuda(), XM.cuda(), X.cuda()
    yR, yM, y = yR.cuda(), yM.cuda(), y.cuda()
    yhR, yhM, yh = yhR.cuda(), yhM.cuda(), yh.cuda()

# %% [markdown]
# ## FNP

# %%
torch.manual_seed(5)


# %%
fnp = RegressionFNP(dim_x=1, dim_y=1, transf_y=stdy, dim_h=100, dim_u=3, n_layers=1, num_M=XM.size(0), 
                    dim_z=50, fb_z=1.0, use_plus=False)

if torch.cuda.is_available():
    fnp = fnp.cuda()

fnp=train_model(fnp, epochs=10000, lr=1e-4)

# %% [markdown]
# ## FNP+

# %%
fnpp = RegressionFNP(dim_x=1, dim_y=1, transf_y=stdy, dim_h=100, dim_u=3, n_layers=1, num_M=XM.size(0), 
                    dim_z=50, fb_z=1.0, use_plus=True)
fnpp = train_model(fnpp, epochs=10000, lr=1e-4)

# %% [markdown]
# ## Attention FNP

# %%
attt = PoolingFNP(dim_x=1, dim_y=1, transf_y=None, dim_h=100, dim_u=3, n_layers=1, num_M=XM.size(0), 
                    dim_z=50, fb_z=1.0, use_plus=False, use_direction_mu_nu=True, set_transformer=True)
attt = train_model(attt, epochs=10000, lr=1e-4)

# %% [markdown]
# ## Deep-set FNP

# %%
poolnp = PoolingFNP(dim_x=1, dim_y=1, transf_y=stdy, dim_h=100, dim_u=3, n_layers=1, num_M=XM.size(0), 
                    dim_z=50, fb_z=0.5, use_plus=False, use_direction_mu_nu=True, set_transformer=False)
poolnp = train_model(poolnp, epochs=10000, lr=1e-4)


