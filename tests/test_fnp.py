import pytest

import torch
import numpy as np
from torch.optim import Adam
from sklearn.preprocessing import StandardScaler

from bliss.models.fnp import (
    DepGraph,
    FNP,
    SequentialVarg,
    MLP,
    SplitLayer,
    NormalEncoder,
    AveragePooler,
    SetPooler,
    RepEncoder,
    ConcatLayer,
)


class TestFNP:
    def test_fnp_onedim(self, paths):
        # One dimensional example
        od = OneDimDataset()
        vanilla_fnp = make_onedim_model()
        fnpp = make_onedim_model(use_plus=True)
        attt = make_onedim_model(use_set_pooler=True, use_attention=True)
        poolnp = make_onedim_model(use_set_pooler=True, use_attention=False, fb_z=0.5)
        model_names = ["fnp", "fnp plus", "pool - attention", "pool - deep set"]
        thresholds = [0.5, 0.75, 0.5, 0.75]
        for (i, model) in enumerate([vanilla_fnp, fnpp, attt, poolnp]):
            print(model_names[i])
            if torch.cuda.is_available():
                od.cuda()
                model = model.cuda()
            model, loss_initial, loss_final, loss_best = train_onedim_model(
                model, od, epochs=1000, lr=1e-4
            )
            ## These are to flag if the model has changed sufficiently to
            ## have a much different starting loss value
            assert loss_initial < 70
            assert loss_initial > 40

            assert loss_best < loss_initial * thresholds[i]
            # Smoke test for prediction
            pred = model.predict(od.XM, od.XR, od.yR[0].unsqueeze(0))


## Onedim example
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


def train_onedim_model(model, od, epochs=10000, lr=1e-4):
    if torch.cuda.is_available():
        model = model.cuda()
    optimizer = Adam(model.parameters(), lr=lr)
    model.train()
    holdout_loss_prev = np.infty
    holdout_loss_initial = model(od.XR, od.yhR, od.XM, od.yhM)
    holdout_loss_best = holdout_loss_initial
    print("Initial holdout loss: {:.3f})".format(holdout_loss_initial.item()))
    for i in range(epochs):
        optimizer.zero_grad()

        loss = model(od.XR, od.yR, od.XM, od.yM)
        loss.backward()
        optimizer.step()

        if i % int(epochs / 10) == 0:
            print("Epoch {}/{}, loss: {:.3f}".format(i, epochs, loss.item()))
            holdout_loss = model(od.XR, od.yhR, od.XM, od.yhM)
            if holdout_loss < holdout_loss_best:
                holdout_loss_best = holdout_loss
            print("Holdout loss: {:.3f}".format(holdout_loss.item()))
    print("Done.")
    return model, holdout_loss_initial, holdout_loss, holdout_loss_best
