import pytest

import torch
import bliss.models.fnp as fnp
import os
import math
import numpy as np
import torch.nn as nn
from matplotlib import pyplot as plt
from torch.optim import Adam
from sklearn.preprocessing import StandardScaler
from torch.distributions import MultivariateNormal

from bliss.models.fnp import (
    DepGraph,
    FNP,
    SequentialVarg,
    MLP,
    Conv2DAutoEncoder,
    SplitLayer,
    NormalEncoder,
    IdentityEncoder,
    AveragePooler,
    SetPooler,
    RepEncoder,
    ConcatLayer,
    ReshapeWrapper,
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
        star_data = PsfFnpData(
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

        fnp_model = ConvPoolingFNP(
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


## Rotation example


class ConvPoolingFNP(FNP):
    def __init__(
        self,
        dim_x=1,
        dim_y=1,
        dim_h=50,
        transf_y=None,
        n_layers=1,
        use_plus=True,
        dim_u=1,
        dim_z=1,
        fb_z=0.0,
        y_encoder_layers=[128],
        mu_nu_layers=[128],
        use_x_mu_nu=True,
        use_direction_mu_nu=False,
        output_layers=[128],
        x_as_u=False,
        pooler=None,
        pooling_layers=[64],
        pooling_rep_size=32,
        set_transformer=False,
        st_numheads=[2, 2],
        size_h=10,
        size_w=10,
        kernel_sizes=[3, 3],
        strides=[1, 1],
        conv_channels=[20, 20],
    ):
        dim_u = dim_u if not x_as_u else dim_x
        if not x_as_u:
            cov_vencoder = SequentialVarg(
                MLP(dim_x, [dim_h] * n_layers, 2 * dim_u),
                SplitLayer(dim_u, -1),
                NormalEncoder(),
            )
        else:
            cov_vencoder = IdentityEncoder()
        dep_graph = DepGraph(dim_u)
        # p(u|x)
        # q(z|x)
        # See equation 7
        # self.q_z = nn.Linear(self.dim_h, 2 * self.dim_z)
        # for p(z|A, XR, yR)

        dim_y_enc = 2 * dim_z
        trans_cond_y = MLP(
            dim_y,
            y_encoder_layers,
            2 * dim_z,
        )
        mu_nu_in = dim_y_enc
        if use_x_mu_nu is True:
            mu_nu_in += dim_x
        if use_direction_mu_nu:
            mu_nu_in += 1
        # self.mu_nu_in = mu_nu_in
        mu_nu_theta = MLP(mu_nu_in, mu_nu_layers, 2 * dim_z)
        rep_encoder = RepEncoder(mu_nu_theta, use_u_diff=False, use_x=use_x_mu_nu)
        if pooler is None:
            pooler = SequentialVarg(
                AveragePooler(dim_z),
                SplitLayer(dim_z, -1),
                NormalEncoder(minscale=1e-8),
            )
        prop_inputs = [1]
        prop_mlp_in = dim_y_enc
        if use_x_mu_nu:
            prop_inputs += [0]
            prop_mlp_in += dim_x
        prop_vencoder = SequentialVarg(
            ConcatLayer(prop_inputs),
            MLP(
                prop_mlp_in,
                mu_nu_layers,
                2 * dim_z,
            ),
            SplitLayer(dim_z, -1),
            NormalEncoder(minscale=1e-8),
        )
        # for p(y|z)
        output_inputs = [0] if not use_plus else [0, 1]
        output_insize = dim_z if not use_plus else dim_z + dim_u
        label_vdecoder = SequentialVarg(
            ConcatLayer(output_inputs),
            MLP(output_insize, output_layers, 2 * dim_y),
            SplitLayer(dim_y, -1),
            NormalEncoder(minscale=0.1),
        )
        super().__init__(
            cov_vencoder,
            dep_graph,
            trans_cond_y,
            rep_encoder,
            pooler,
            prop_vencoder,
            label_vdecoder,
            fb_z=fb_z,
        )
        self.transf_y = transf_y
        # dim_z = kwargs["dim_z"]
        # dim_u = kwargs["dim_u"]
        # mu_nu_layers = kwargs.get("mu_nu_layers", [128])
        dim_y_enc = 2 * dim_z
        mu_nu_in = dim_y_enc + dim_u
        mu_nu_theta = MLP(mu_nu_in, mu_nu_layers, pooling_rep_size)
        self.rep_encoder = RepEncoder(mu_nu_theta, use_u_diff=True, use_x=False)

        self.pooler = SequentialVarg(
            SetPooler(
                mu_nu_theta.out_features,
                dim_z,
                pooling_layers,
                set_transformer,
                st_numheads,
            ),
            SplitLayer(dim_z, -1),
            NormalEncoder(minscale=1e-8),
        )

        conv_autoencoder = Conv2DAutoEncoder(
            size_h,
            size_w,
            conv_channels,
            kernel_sizes,
            strides,
        )

        self.trans_cond_y = nn.Sequential(
            ReshapeWrapper(conv_autoencoder.encoder, k=2),
            nn.Linear(conv_autoencoder.dim_rep, conv_autoencoder.dim_rep),
        )
        dim_y_enc = conv_autoencoder.dim_rep
        mu_nu_in = dim_y_enc
        if use_x_mu_nu is True:
            mu_nu_in += dim_x
        if use_direction_mu_nu:
            mu_nu_in += 1
        # self.mu_nu_in = mu_nu_in
        ## This is a quick fix, but pooling_rep_size should not be used
        mu_nu_theta = MLP(mu_nu_in, mu_nu_layers, pooling_rep_size)
        self.rep_encoder = RepEncoder(mu_nu_theta, use_u_diff=True, use_x=False)
        prop_inputs = [1]
        prop_mlp_in = dim_y_enc
        if use_x_mu_nu:
            prop_inputs += [0]
            prop_mlp_in += dim_x
        self.prop_vencoder = SequentialVarg(
            ConcatLayer(prop_inputs),
            MLP(prop_mlp_in, mu_nu_layers, 2 * dim_z),
            SplitLayer(dim_z, -1),
            NormalEncoder(minscale=1e-8),
        )
        output_inputs = [0] if not use_plus else [0, 1]
        output_insize = dim_z if not use_plus else dim_z + dim_u
        self.label_vdecoder = SequentialVarg(
            ConcatLayer(output_inputs),
            MLP(output_insize, output_layers, conv_autoencoder.dim_rep),
            ReshapeWrapper(conv_autoencoder.decoder, k=2),
            SplitLayer(1, -3),
            NormalEncoder(minscale=0.1),
        )

        self.pooler = SequentialVarg(
            SetPooler(
                mu_nu_theta.out_features,
                dim_z,
                pooling_layers,
                set_transformer,
                st_numheads,
            ),
            SplitLayer(dim_z, -1),
            NormalEncoder(minscale=1e-8),
        )

    def predict(self, x_new, XR, yR, sample=True, A_in=None, sample_Z=True):
        y_pred = super().predict(
            x_new, XR, yR, sample=sample, A_in=A_in, sample_Z=sample_Z
        )
        if self.transf_y is not None:
            y_pred = self.inverse_transform(y_pred)
        return y_pred

    def inverse_transform(self, y):
        y = y.squeeze(-3)
        y_flat = y.reshape(*y.shape[0:2], -1).cpu().detach().numpy()
        y_flat_invt = self.transf_y.inverse_transform(y_flat)
        y_out = torch.from_numpy(y_flat_invt).resize_(y.size())
        return y_out


class PSFRotate:
    """
    Class for generating synthetic sequences of stars which rotate.
    """

    def __init__(
        self,
        X,
        size_h=10,
        size_w=10,
        base_angle=math.pi / 100,
        angle_stdev=math.pi / 300,
        cov_multiplier=6.0,
        bright_val=3.0,
        bright_skip=5,
        star_width=0.25,
    ):
        """
        :param X: Locations of the stars
        :param size_h: The height of each image in pixels
        :param size_w: The width of each image in pixels
        :param base_angle: How much the star rotates on average in radians per unit of X
        :param angle_stdev: The standard deviation of rotation (flat; not per unit of X!)
        :param cov_multiplier: How much covariance the log normal density serving as the PSF will have
        :param bright_val: How much brighter are bright stars
        :param bright_skip: Which stars are bright
        :param star_width: The width of the star (lower values lead to skinnier stars)
        """

        self.X = X
        self.size_h = size_h
        self.size_w = size_w
        self.base_angle = base_angle
        self.angle_stdev = angle_stdev
        self.cov_multiplier = cov_multiplier
        self.bright_val = bright_val
        self.bright_skip = bright_skip
        self.star_width = star_width

        self.base_cov = (
            torch.tensor([[1.0, 0.0], [0.0, self.star_width]]) * self.cov_multiplier
        )
        self.I = self.X.size(0)

        self.h = torch.tensor(range(self.size_h), dtype=torch.float32)
        self.w = torch.tensor(range(self.size_w), dtype=torch.float32)
        self.grid = torch.stack(
            [
                self.h.unsqueeze(1).repeat(1, self.size_w),
                self.w.unsqueeze(0).repeat(self.size_h, 1),
            ],
            2,
        )

    def generate(self, N, device=None):
        start = torch.rand(N, device=device) * math.pi
        eps = torch.randn(N, self.I, device=device) * self.angle_stdev
        phi = start.unsqueeze(1) + (self.X.to(device).unsqueeze(0) + eps) * (
            self.base_angle
        )

        idx_brights = (
            torch.fmod(torch.tensor(range(self.I), device=device), self.bright_skip)
            == 0
        )
        l = torch.ones(N, self.I, device=device)
        l[:, idx_brights] = self.bright_val

        mu = torch.tensor([torch.mean(self.h), torch.mean(self.w)], device=device)
        rots = self.make_rot_matrices(phi)
        covs = (
            rots.transpose(3, 2)
            .matmul(self.base_cov.to(device))
            .matmul(rots)
            .unsqueeze(2)
            .unsqueeze(2)
        )
        pixel_dist = MultivariateNormal(mu, covs)

        brights = pixel_dist.log_prob(
            self.grid.to(device).unsqueeze(0)
        ).exp() * l.unsqueeze(-1).unsqueeze(-1)
        return brights

    @staticmethod
    def make_rot_matrices(phis):
        cos_phi = phis.cos()
        sin_phi = phis.sin()
        row1 = torch.stack([cos_phi, -sin_phi], -1)
        row2 = torch.stack([sin_phi, cos_phi], -1)
        y = torch.stack([row1, row2], -1)
        return y


class PsfFnpData:
    def __init__(self, n_ref, Ks, N, N_valid=None, conv=False, device=None, **kwargs):
        self.n_ref = n_ref
        self.Ks = Ks
        self.N = N
        self.conv = conv
        self.device = device

        if N_valid is None:
            self.N_valid = self.N
        else:
            self.N_valid = N_valid

        self.X_ref, self.X_dep, self.X_all, self.idx_ref, self.idx_dep = self.make_X(
            n_ref, Ks
        )
        self.G, self.A = self.make_graphs(n_ref, Ks)

        self.dgp = PSFRotate(self.X_all, **kwargs)

        self.images, self.stdx, self.stdy, X, y = self.generate(self.N)
        (
            self.X_r,
            self.y_r,
            self.X_m,
            self.y_m,
            self.X,
            self.y,
        ) = self.split_reference_dependent(X, y)

        self.images_valid, _, _, X, y = self.generate(
            self.N_valid, self.stdx, self.stdy
        )
        (
            self.X_r_valid,
            self.y_r_valid,
            self.X_m_valid,
            self.y_m_valid,
            self.X_valid,
            self.y_valid,
        ) = self.split_reference_dependent(X, y)

    @staticmethod
    def make_graph_ref_pair(j1, j2, K, n_ref):
        """
        This calculates dependency matrix for points between j1 and j2
        :param j1: Index of first reference point
        :param j2: Index of second reference point
        :param K: Number of dependent points
        :param n_ref: Number of reference_points
        """
        X = torch.zeros(K, n_ref)
        X[:, j1] = 1
        X[:, j2] = 1
        return X

    def make_graphs(self, n_ref, Ks):
        """
        This calculates the dependency matrices G and A, assuming that there are Ks[j] interpolated points between
        references j and j+1
        :param n_ref: Number of reference_points
        :param Ks: List of integers indicated number of interpolated points in that gap
        """
        G = torch.zeros(n_ref, n_ref, dtype=torch.float32)
        for j in range(n_ref - 1):
            G[j + 1, j] = 1.0

        A = torch.tensor([], dtype=torch.float32)
        for j in range(n_ref - 1):
            A = torch.cat([A, self.make_graph_ref_pair(j, j + 1, Ks[j], n_ref)])

        return G, A

    def generate(self, N, stdx=None, stdy=None):
        images = self.dgp.generate(N, device=self.device).cpu()
        Xmat = self.X_all.unsqueeze(1)
        ymat = images.reshape(N, self.dgp.I, -1)

        if stdx is None:
            stdx = StandardScaler().fit(Xmat)
        if stdy is None:
            stdy = StandardScaler().fit(ymat.reshape(N * self.dgp.I, -1))
        X, y = (
            stdx.transform(Xmat),
            stdy.transform(ymat.reshape(N * self.dgp.I, -1)).reshape(N, self.dgp.I, -1),
        )

        return images, stdx, stdy, X, y

    def split_reference_dependent(self, X, y):
        idxR = self.idx_ref
        idxM = self.idx_dep
        N = y.shape[0]

        X_r = torch.from_numpy(X[idxR, :].astype(np.float32))
        y_r = torch.from_numpy(y[:, idxR, :].astype(np.float32))
        X_m = torch.from_numpy(X[idxM, :].astype(np.float32))
        y_m = torch.from_numpy(y[:, idxM, :].astype(np.float32))
        X = torch.from_numpy(X.astype(np.float32))
        y = torch.from_numpy(y.astype(np.float32))

        if self.conv:
            y_r = y_r.reshape(N, self.n_ref, 1, self.dgp.size_h, self.dgp.size_w)
            y_m = y_m.reshape(
                N, self.dgp.I - self.n_ref, 1, self.dgp.size_h, self.dgp.size_w
            )
            y = y.reshape(N, self.dgp.I, 1, self.dgp.size_h, self.dgp.size_w)

        return X_r, y_r, X_m, y_m, X, y

    def markref(self, img, max_bright=None):
        if max_bright is None:
            max_bright = self.images.max()
        img[0, :] = max_bright
        img[self.dgp.size_h - 1, :] = max_bright
        img[:, 0] = max_bright
        img[:, self.dgp.size_w - 1] = max_bright
        return img

    @staticmethod
    def make_X_ref_pair(x1, x2, K):
        X = x1 + (x2 - x1) / (K + 1) * (
            torch.tensor(range(K), dtype=torch.float32) + 1.0
        )
        return X

    def make_X(self, n_ref, Ks):
        X_ref = torch.tensor(range(0, n_ref * 2, 2), dtype=torch.float32)
        X_dep = torch.tensor([], dtype=torch.float32)
        for i in range(n_ref - 1):
            X_dep = torch.cat(
                [X_dep, self.make_X_ref_pair(X_ref[i], X_ref[i + 1], Ks[i])]
            )
        X_all, idxs = torch.cat([X_ref, X_dep], 0).sort()

        Is = torch.tensor(range(X_all.size(0)))
        idx_ref = Is[idxs < n_ref]
        idx_dep = Is[idxs >= n_ref]
        return X_ref, X_dep, X_all, idx_ref, idx_dep

    def export_images(self, path, mark_ref=True, valid=False, nrows=None):
        if valid:
            images = self.images_valid
        else:
            images = self.images
        if nrows is None:
            nrows = self.N
        vmin = self.images.min()
        vmax = self.images.max()
        image_lng = torch.tensor([])
        for n in range(nrows):
            row = torch.tensor([])
            for i in range(self.dgp.I):
                img = images[n, i]
                if mark_ref and (i in self.idx_ref):
                    img = self.markref(img)
                row = torch.cat([row, img], dim=1)
            image_lng = torch.cat([image_lng, row], dim=0)
        plt.imsave(path, image_lng, vmin=vmin, vmax=vmax)

    def cuda(self):
        self.X_r, self.X_m, self.X = self.X_r.cuda(), self.X_m.cuda(), self.X.cuda()
        self.y_r, self.y_m, self.y = self.y_r.cuda(), self.y_m.cuda(), self.y.cuda()
        self.X_r_valid, self.X_m_valid, self.X_valid = (
            self.X_r_valid.cuda(),
            self.X_m_valid.cuda(),
            self.X_valid.cuda(),
        )
        self.y_r_valid, self.y_m_valid, self.y_valid = (
            self.y_r_valid.cuda(),
            self.y_m_valid.cuda(),
            self.y_valid.cuda(),
        )
        self.A = self.A.cuda()
        self.G = self.G.cuda()

    def cpu(self):
        self.X_r, self.X_m, self.X = self.X_r.cpu(), self.X_m.cpu(), self.X.cpu()
        self.y_r, self.y_m, self.y = self.y_r.cpu(), self.y_m.cpu(), self.y.cpu()
        self.A = self.A.cpu()
        self.G = self.G.cpu()

    def predict_n(self, y_r, fnp_model, X=None, A=None, sample_Z=True):
        """
        Make a prediction using the generating X_dep
        :param n: The index of the Ys to use
        :param fnp_model: Trained RegressionFNP
        """
        if X is None:
            X = self.X_m
        if A is None:
            A = self.A.cuda()
        pred_np = fnp_model.predict(X, self.X_r, y_r, A_in=A, sample_Z=sample_Z)
        # pred    = torch.from_numpy(pred_np)
        # newsize = torch.Size([pred.size(0), self.dgp.size_h, self.dgp.size_w])
        return pred_np[0]

    def quantiles_n(self, y_r, fnp_model, quantiles=[0.05, 0.95], samples=1000):
        preds = []
        for i in range(samples):
            preds.append(self.predict_n(y_r, fnp_model))
        pred_tens = torch.stack(preds)
        quant_out = []
        for q in quantiles:
            quant_out.append(
                torch.from_numpy(np.percentile(pred_tens, q, axis=0)).to(torch.float32)
            )

        return quant_out

    def mean_n(self, y_r, fnp_model, X=None, A=None, samples=1000, sample_Z=True):
        preds = []
        for i in range(samples):
            preds.append(self.predict_n(y_r, fnp_model, X=X, A=A, sample_Z=sample_Z))
        pred_tens = torch.stack(preds)
        return pred_tens.mean(dim=0)

    def make_fnp_pred_image_n(self, X_dep, Y_d, n, valid=True):
        if valid:
            images = self.images_valid
        else:
            images = self.images
        X_all, idxs = torch.cat([self.X_ref, X_dep]).sort()
        Is = torch.tensor(range(X_all.size(0)))
        idxs_ref = Is[idxs < self.n_ref]
        idxs_dep = Is[idxs >= self.n_ref]

        max_bright = torch.max(Y_d)

        img = torch.tensor([])

        for i in Is:
            if i in idxs_ref:
                img_i = self.markref(images[n, i, :, :])
            else:
                img_i = Y_d[idxs_dep == i, :].squeeze(0)
            img = torch.cat([img, img_i], dim=1)

        return img

    def make_fnp_pred_image(self, fnp_model):
        imgs = []
        for i in range(self.N):
            predi = self.predict_n(i, fnp_model)
            imgi = self.make_fnp_pred_image_n(self.X_dep, predi, i)
            imgs.append(imgi)

        myimg = torch.cat(imgs, 0)
        return myimg

    def make_fnp_quant_image(
        self, fnp_model, Ns=None, quantiles=[0.05, 0.95], samples=1000, valid=True
    ):
        if Ns is None:
            Ns = range(self.N)

        quants = []
        for i in Ns:
            quanti = self.quantiles_n(
                i, fnp_model, quantiles=quantiles, samples=samples
            )
            quants.append(quanti)

        quantimgs = []
        for j in range(len(quantiles)):
            imgs = []
            for i in Ns:
                predi = quants[i][j]
                imgi = self.make_fnp_pred_image_n(self.X_dep, predi, i, valid=valid)
                imgs.append(imgi)
            my_img = torch.cat(imgs, 0)
            quantimgs.append(my_img)

        return quantimgs

    def make_fnp_mean_image(
        self,
        fnp_model,
        X=None,
        X_nostd=None,
        A=None,
        N=None,
        samples=1000,
        valid=True,
        sample_Z=True,
    ):
        imgs = []
        if N is None:
            if valid:
                N = self.N_valid
            else:
                N = self.N
        for i in range(N):
            if valid:
                y_r = self.y_r_valid[i : (i + 1)]
            else:
                y_r = self.y_r[i : (i + 1)]
            predi = self.mean_n(
                y_r, fnp_model, X=X, A=A, samples=samples, sample_Z=sample_Z
            )
            if X_nostd is None:
                X_dep = self.X_dep.cpu()
            else:
                X_dep = X_nostd.cpu()
            imgi = self.make_fnp_pred_image_n(X_dep, predi, i, valid=valid)
            imgs.append(imgi)

        myimg = torch.cat(imgs, 0)
        return myimg

    def make_fnp_single_image(self, fnp_model, N=None, valid=True, sample_Z=True):
        return self.make_fnp_mean_image(
            fnp_model, N=N, samples=1, valid=valid, sample_Z=sample_Z
        )

    def make_fnp_var_image(self, fnp_model, N=None, samples=1000, valid=True):
        imgs = []
        if N is None:
            if valid:
                N = self.N_valid
            else:
                N = self.N
        for i in range(N):
            if valid:
                y_r = self.y_r_valid[i : (i + 1)]
            else:
                y_r = self.y_r[i : (i + 1)]
            predi = self.mean_n(y_r.pow(2), fnp_model, samples=samples) - self.mean_n(
                y_r, fnp_model, samples=samples
            ).pow(2)
            imgi = self.make_fnp_pred_image_n(self.X_dep, predi, i, valid=valid)
            imgs.append(imgi)

        myimg = torch.cat(imgs, 0)
        return myimg
