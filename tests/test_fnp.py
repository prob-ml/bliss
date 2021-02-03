import pytest

import torch
import bliss.models.fnp as fnp
import os
import math
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
    RepEncoder,
    ConcatLayer,
)
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
def train_onedim_model(model, od, epochs=10000, lr=1e-4):
    if torch.cuda.is_available():
        model = model.cuda()
    optimizer = Adam(model.parameters(), lr=lr)
    model.train()
    holdout_loss_prev = np.infty
    holdout_loss_initial = model(od.XR, od.yhR, od.XM, od.yhM)
    holdout_loss_best = holdout_loss_initial
    print("Initial holdout loss: {:.3f})".format(holdout_loss_initial.item()))
    if isinstance(model, fnp.RegressionFNP):
        stdy = None
    else:
        stdy = od.stdy
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

class TestFNP:
    def test_fnp_onedim(self, paths):
        # One dimensional example
        od = OneDimDataset()
        dim_x = 1
        dim_y = 1
        dim_z = 50
        dim_u = 3
        dim_y_enc = 100
        vanilla_fnp = FNP(
            cov_vencoder=SequentialVarg(
                MLP(dim_x, [100], 2 * dim_u),
                SplitLayer(dim_u, -1),
                NormalEncoder(),
            ),
            dep_graph=DepGraph(dim_u),
            trans_cond_y=MLP(dim_y, [128], dim_y_enc),
            rep_encoder=RepEncoder(
                MLP(dim_y_enc + dim_x, [128], 2 * dim_z), use_u_diff=False, use_x=True
            ),
            pooler=SequentialVarg(
                AveragePooler(dim_z),
                SplitLayer(dim_z, -1),
                NormalEncoder(minscale=1e-8),
            ),
            prop_vencoder=SequentialVarg(
                ConcatLayer([1, 0]),
                MLP(
                    dim_y_enc + dim_x,
                    [128],
                    2 * dim_z,
                ),
                SplitLayer(dim_z, -1),
                NormalEncoder(minscale=1e-8),
            ),
            label_vdecoder=SequentialVarg(
                ConcatLayer([0]),
                MLP(dim_z, [128], 2 * dim_y),
                SplitLayer(dim_y, -1),
                NormalEncoder(minscale=0.1),
            ),
            fb_z=1.0,
        )

        fnpp = fnp.RegressionFNP(
            dim_x=1,
            dim_y=1,
            transf_y=od.stdy,
            dim_h=100,
            dim_u=3,
            n_layers=1,
            dim_z=50,
            fb_z=1.0,
            use_plus=True,
        )

        attt = fnp.PoolingFNP(
            dim_x=1,
            dim_y=1,
            transf_y=None,
            dim_h=100,
            dim_u=3,
            n_layers=1,
            dim_z=50,
            fb_z=1.0,
            use_plus=False,
            use_direction_mu_nu=True,
            set_transformer=True,
        )

        poolnp = fnp.PoolingFNP(
            dim_x=1,
            dim_y=1,
            transf_y=od.stdy,
            dim_h=100,
            dim_u=3,
            n_layers=1,
            dim_z=50,
            fb_z=0.5,
            use_plus=False,
            use_direction_mu_nu=True,
            set_transformer=False,
        )

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

    def test_fnp_rotate(self, paths):
        ## Rotating star

        if torch.get_num_threads() > 8:
            torch.set_num_threads(8)

        # print("Setting CUDA device to", args.cuda_device[0])
        # torch.cuda.set_device(args.cuda_device[0])

        epochs_outer = 100
        # generate = True
        n_ref = 10
        N = 10
        size_h = 50
        size_w = 50
        cov_multiplier = 20.0
        base_angle = math.pi * (3.0 / 23.0)
        angle_stdev = math.pi * 0.025

        K_probs = torch.tensor([0.25] * 4)
        Ks = torch.multinomial(K_probs, n_ref - 1, replacement=True) + 1

        batch_size = 1
        learnrate = 0.001

        print("Generating data...")
        star_data = fnp.PsfFnpData(
            n_ref,
            Ks,
            N,
            size_h=size_h,
            size_w=size_w,
            cov_multiplier=cov_multiplier,
            base_angle=base_angle,
            angle_stdev=angle_stdev,
            bright_skip=1,
            star_width=0.1,
            conv=True,
            N_valid=10,
            device="cpu",
        )
        star_data.export_images("TEST_rotating_star_dgp.png", nrows=10)
        star_data.export_images(
            "TEST_rotating_star_dgp_valid.png", nrows=10, valid=True
        )
        os.remove("TEST_rotating_star_dgp.png")
        os.remove("TEST_rotating_star_dgp_valid.png")

        fnp_model = fnp.ConvPoolingFNP(
            dim_x=1,
            size_h=size_h,
            size_w=size_w,
            kernel_sizes=[3, 3, 3, 3, 3],
            strides=[1, 2, 2, 2, 2],
            conv_channels=[16, 32, 64, 128, 256],
            transf_y=star_data.stdy,
            dim_h=100,
            dim_u=1,
            pooling_layers=[32, 16],
            n_layers=3,
            dim_z=8,
            fb_z=1.0,
            use_plus=False,
            mu_nu_layers=[128, 64, 32],
            use_x_mu_nu=False,
            use_direction_mu_nu=True,
            output_layers=[32, 64, 128],
            x_as_u=True,
            set_transformer=True,
        )

        if torch.cuda.is_available():
            star_data.cuda()
            fnp_model = fnp_model.cuda()

        optimizer = Adam(fnp_model.parameters(), lr=learnrate)
        fnp_model.train()

        loss_initial = (
            fnp_model(
                star_data.X_r_valid,
                star_data.y_r_valid[[0]],
                star_data.X_m_valid,
                star_data.y_m_valid[[0]],
                G_in=star_data.G,
                A_in=star_data.A,
            )
            / star_data.N_valid
        )
        y_r_in = star_data.y_r_valid[[2]]
        y_m_in = star_data.y_m_valid[[2]]
        loss_initial = (
            fnp_model(
                star_data.X_r_valid,
                y_r_in,
                star_data.X_m_valid,
                y_m_in,
                G_in=star_data.G,
                A_in=star_data.A,
            )
            / batch_size
        )
        print("Initial loss : {:.3f}".format(loss_initial.item()))
        callback = 100
        epochs = epochs_outer * star_data.N
        nbatches = star_data.N // batch_size
        for i in range(epochs_outer):
            points = torch.randperm(star_data.N)
            for i_n in range(nbatches):
                ns = points[(batch_size * i_n) : (batch_size * (i_n + 1))]
                optimizer.zero_grad()
                y_r_in = star_data.y_r[ns]
                y_m_in = star_data.y_m[ns]
                loss = (
                    fnp_model(
                        star_data.X_r,
                        y_r_in,
                        star_data.X_m,
                        y_m_in,
                        G_in=star_data.G,
                        A_in=star_data.A,
                    )
                    / batch_size
                )
                loss.backward()
                optimizer.step()
                if ((i * N) + i_n) % int(epochs / callback) == 0:
                    print(
                        "({:05.1f}%)Epoch {}/{}. loss: {:.3f}".format(
                            ((i * N) + i_n) / epochs * 100,
                            (i * N) + i_n,
                            epochs,
                            loss.item(),
                        )
                    )

        loss_final = (
            fnp_model(
                star_data.X_r_valid,
                star_data.y_r_valid[[0]],
                star_data.X_m_valid,
                star_data.y_m_valid[[0]],
                G_in=star_data.G,
                A_in=star_data.A,
            )
            / star_data.N_valid
        )
        print("Final loss : {:.3f}".format(loss_final.item()))

        assert loss_initial < 7000
        assert loss_initial > 4000
        assert loss_final < loss_initial - 5000
