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
    RegressionFNP,
    PoolingFNP,
    OneDimDataset,
    train_onedim_model,
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
od = OneDimDataset()
# %%
if torch.cuda.is_available():
    od.cuda()

# %% [markdown]
# ## FNP

# %%
torch.manual_seed(5)


# %%
fnp = RegressionFNP(
    dim_x=1,
    dim_y=1,
    transf_y=od.stdy,
    dim_h=100,
    dim_u=3,
    n_layers=1,
    num_M=od.XM.size(0),
    dim_z=50,
    fb_z=1.0,
    use_plus=False,
)

if torch.cuda.is_available():
    fnp = fnp.cuda()

fnp, _, fnp_loss_final = train_onedim_model(
    fnp, od, epochs=10000, lr=1e-4, visualize=VISUALIZE
)

# %% [markdown]
# ## FNP+

# %%
fnpp = RegressionFNP(
    dim_x=1,
    dim_y=1,
    transf_y=od.stdy,
    dim_h=100,
    dim_u=3,
    n_layers=1,
    num_M=od.XM.size(0),
    dim_z=50,
    fb_z=1.0,
    use_plus=True,
)
fnpp, _, fnpp_loss_final = train_onedim_model(
    fnpp, od, epochs=10000, lr=1e-4, visualize=VISUALIZE
)

# %% [markdown]
# ## Attention FNP

# %%
attt = PoolingFNP(
    dim_x=1,
    dim_y=1,
    transf_y=None,
    dim_h=100,
    dim_u=3,
    n_layers=1,
    num_M=od.XM.size(0),
    dim_z=50,
    fb_z=1.0,
    use_plus=False,
    use_direction_mu_nu=True,
    set_transformer=True,
)
attt, _, attt_loss_final = train_onedim_model(
    attt, od, epochs=10000, lr=1e-4, visualize=VISUALIZE
)

# %% [markdown]
# ## Deep-set FNP

# %%
poolnp = PoolingFNP(
    dim_x=1,
    dim_y=1,
    transf_y=od.stdy,
    dim_h=100,
    dim_u=3,
    n_layers=1,
    num_M=od.XM.size(0),
    dim_z=50,
    fb_z=0.5,
    use_plus=False,
    use_direction_mu_nu=True,
    set_transformer=False,
)
poolnp, _, poolnp_loss_final = train_onedim_model(
    poolnp, od, epochs=10000, lr=1e-4, visualize=VISUALIZE
)
