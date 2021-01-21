import torch.nn as nn
import torch
import math
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np
from torch.distributions import Bernoulli
from itertools import product

# from .utils import Normal, L1Error, float_tensor, logitexp, sample_DAG, sample_bipartite, Flatten, UnFlatten, one_hot, norm_graph_weighted, ResidualLayer
# from .set_transformer.modules import SAB, PMA
from torch.distributions import Categorical, Bernoulli
from torch.optim import Adam


def make_fc_net(insize, hs, outsize, act=nn.ReLU, final=None, resid=False):
    layers = [nn.Sequential(nn.Linear(insize, hs[0]), act())]
    for i in range(len(hs) - 1):
        layers.append(nn.Sequential(nn.Linear(hs[i], hs[i + 1]), act()))
        if resid:
            layers[-1] = ResidualLayer(layers[-1])
    layers.append(nn.Linear(hs[-1], outsize))
    # if resid:
    #     layers[-1] = ResidualLayer(layers[-1])
    if final is not None:
        layers += [final()]
    return nn.Sequential(*layers)


class SkipConnection(nn.Module):
    def __init__(self, f, size_skip):
        super(SkipConnection, self).__init__()
        self.f = f
        self.size_skip = size_skip

    def forward(self, x):
        y = self.f(x)
        _, x_skip = torch.split(x, [x.size(-1) - self.size_skip, self.size_skip], -1)
        return torch.cat([y, x_skip], -1)


def linear_skip_connection(featin, featout, size_skip=1):
    return SkipConnection(nn.Linear(featin, featout), size_skip)


def make_fc_skip(
    insize, hs, outsize, act=nn.ReLU, final=None, size_skip=1, skip_final=False
):
    layers = [linear_skip_connection(insize, hs[0], size_skip=size_skip), act()]
    for i in range(len(hs) - 1):
        layers += [
            linear_skip_connection(hs[i] + size_skip, hs[i + 1], size_skip=size_skip),
            act(),
        ]
    if skip_final:
        layers += [
            linear_skip_connection(hs[-1] + size_skip, outsize, size_skip=size_skip)
        ]
    else:
        layers += [nn.Linear(hs[-1] + size_skip, outsize)]
    if final is not None:
        layers += [final()]
    return nn.Sequential(*layers)


def calc_pairwise_isright(uM, uR):
    """
    This returns a binary matrix that is one if uM_i < uR_i and zero otherwise
    """
    Z = uM.unsqueeze(1) - uR.unsqueeze(0)
    return (Z < 0).squeeze(2)


def calc_pairwise_dist2(uM, uR):
    """
    This calculates the cross product of dist2
    """
    Z = uM.unsqueeze(1) - uR.unsqueeze(0)
    # return Z
    return Z.pow(2).sum(2).sqrt()


class RegressionFNP(pl.LightningModule):
    """
    Functional Neural Process for regression
    """

    def __init__(
        self,
        dim_x=1,
        dim_y=1,
        dim_h=50,
        transf_y=None,
        n_layers=1,
        use_plus=True,
        num_M=100,
        dim_u=1,
        dim_z=1,
        fb_z=0.0,
        G=None,
        A=None,
        y_encoder_layers=[128],
        mu_nu_layers=[128],
        use_x_mu_nu=True,
        use_direction_mu_nu=False,
        mu_nu_skip=False,
        output_layers=[128],
        x_as_u=False,
        condition_on_ref=False,
        train_separate_proposal=True,
        train_separate_extrapolate=False,
        discrete_orientation=True,
        weighted_graph=False,
        y_encoder_resid=False,
        mu_nu_resid=False,
        output_resid=False,
    ):
        """
        :param dim_x: Dimensionality of the input
        :param dim_y: Dimensionality of the output
        :param dim_h: Dimensionality of the hidden layers
        :param transf_y: Transformation of the output (e.g. standardization)
        :param n_layers: How many hidden layers to us
        :param use_plus: Whether to use the FNP+
        :param num_M: How many points exist in the training set that are not part of the reference set
        :param dim_u: Dimensionality of the latents in the embedding space
        :param dim_z: Dimensionality of the  latents that summarize the parents
        :param fb_z: How many free bits do we allow for the latent variable z
        :param G: Use a supplied G matrix instead of sampling it (default is None)
        :param A: Use a supplied A matrix instead of sampling it (default is None)
        :param y_encoder_layers: Array of integers for the sizes to encode y into z
        :param mu_nu_layers: Array of integers for the network to map encoded x and y to z
        :param use_x_mu_nu: Whether to use x in mu_nu
        :param output_layers: Array of integers for the sizes of the output nn
        """
        super(RegressionFNP, self).__init__()

        # self.num_M = num_M
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.dim_h = dim_h
        self.dim_u = dim_u if not x_as_u else dim_x
        self.dim_z = dim_z
        self.use_plus = use_plus
        self.fb_z = fb_z
        self.transf_y = transf_y
        self.G = G
        self.A = A
        if G is not None:
            self.register_buffer("G_const", self.G)
        if A is not None:
            self.register_buffer("A_const", self.A)
        self.y_encoder_layers = y_encoder_layers
        self.mu_nu_layers = mu_nu_layers
        self.use_x_mu_nu = use_x_mu_nu
        self.use_direction_mu_nu = use_direction_mu_nu
        self.mu_nu_skip = mu_nu_skip
        self.output_layers = output_layers
        self.x_as_u = x_as_u
        self.condition_on_ref = condition_on_ref
        self.train_separate_proposal = train_separate_proposal
        self.train_separate_extrapolate = train_separate_extrapolate
        self.discrete_orientation = discrete_orientation
        self.weighted_graph = weighted_graph
        self.y_encoder_resid = y_encoder_resid
        self.mu_nu_resid = mu_nu_resid
        self.output_resid = output_resid

        # normalizes the graph such that inner products correspond to averages of the parents
        self.norm_graph = lambda x: x / (torch.sum(x, 1, keepdim=True) + 1e-8)

        self.register_buffer("lambda_z", float_tensor(1).fill_(1e-8))

        # function that assigns the edge probabilities in the graph
        self.pairwise_g_logscale = nn.Parameter(
            float_tensor(1).fill_(math.log(math.sqrt(self.dim_u)))
        )
        self.pairwise_g = lambda x: logitexp(
            -0.5
            * torch.sum(
                torch.pow(x[:, self.dim_u :] - x[:, 0 : self.dim_u], 2), 1, keepdim=True
            )
            / self.pairwise_g_logscale.exp()
        ).view(x.size(0), 1)
        # transformation of the input
        init = [nn.Linear(dim_x, self.dim_h), nn.ReLU()]
        for i in range(n_layers - 1):
            init += [nn.Linear(self.dim_h, self.dim_h), nn.ReLU()]
        self.cond_trans = nn.Sequential(*init)
        # p(u|x)
        self.p_u = nn.Linear(self.dim_h, 2 * self.dim_u)
        # q(z|x)
        # See equation 7
        self.q_z = nn.Linear(self.dim_h, 2 * self.dim_z)
        # for p(z|A, XR, yR)
        self.trans_cond_y = self.make_trans_cond_y()

        self.mu_nu_theta = self.make_mu_nu_theta()

        if self.train_separate_proposal:
            self.mu_nu_proposal = self.make_mu_nu_proposal()
        else:
            self.mu_nu_proposal = self.mu_nu_theta
        # for p(y|z)
        self.output = self.make_output()

    def training_step(self, batch, batch_idx):
        yR = batch["YR"]
        yM = batch["YM"]
        XR = batch["XR"][:, 0:1]
        XM = batch["XM"][:, 0:1]

        yR = yR.view(1, yR.size(0), -1)
        yM = yM.view(1, yM.size(0), -1)

        loss = self.forward(XR, yR, XM, yM)
        self.log("train_loss", loss)

        return loss

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=1e-3)
        return optimizer

    def make_trans_cond_y(self):
        self.dim_y_enc = 2 * self.dim_z
        return make_fc_net(
            self.dim_y,
            self.y_encoder_layers,
            2 * self.dim_z,
            resid=self.y_encoder_resid,
        )

    def make_mu_nu_theta(self):
        mu_nu_in = self.dim_y_enc
        if self.use_x_mu_nu is True:
            # raise(NotImplementedError("use_x_mu_nu is not yet implemented")
            mu_nu_in += self.dim_x
        if self.use_direction_mu_nu:
            mu_nu_in += 1

        if self.train_separate_extrapolate:
            mu_nu_in += 1

        return make_fc_net(
            mu_nu_in, self.mu_nu_layers, 2 * self.dim_z, resid=self.mu_nu_resid
        )

    def make_mu_nu_proposal(self):
        mu_nu_in = self.dim_y_enc
        if self.use_x_mu_nu is True:
            # raise(NotImplementedError("use_x_mu_nu is not yet implemented")
            mu_nu_in += self.dim_x
        return make_fc_net(
            mu_nu_in, self.mu_nu_layers, 2 * self.dim_z, resid=self.mu_nu_resid
        )

    def make_output(self):
        output_insize = self.dim_z if not self.use_plus else self.dim_z + self.dim_u
        return make_fc_net(
            output_insize, self.output_layers, 2 * self.dim_y, resid=self.output_resid
        )

    def sample_u(self, X_all, n_ref):
        # get U
        if self.x_as_u:
            u = X_all
        else:
            H_all = self.cond_trans(X_all)
            pu_mean_all, pu_logscale_all = torch.split(
                self.p_u(H_all), self.dim_u, dim=1
            )
            pu = Normal(pu_mean_all, pu_logscale_all)
            u = pu.rsample()
        uR = u[0:n_ref]
        uM = u[n_ref:]
        return u, uR, uM

    def sample_dependency_matrices(self, uR, uM):
        if self.G is None:
            # raise(NotImplementedError("Random graphs are not implemented yet for batch likelihood"))
            G = sample_DAG(uR, self.pairwise_g, training=self.training)
        else:
            G = self.G_const

        if self.A is None:
            # raise(NotImplementedError("Random graphs are not implemented yet for batch likelihood"))
            A = sample_bipartite(uM, uR, self.pairwise_g, training=self.training)
        else:
            A = self.A_const

        return G, A

    def calc_trans_cond_y(self, yR):
        return self.trans_cond_y(yR)

    def calc_pz_dist(self, u, uR, GA, XR, yR, minscale=1e-8):
        yR_encoded = self.calc_trans_cond_y(yR)

        if self.discrete_orientation:
            pairwise_differences = calc_pairwise_isright(u, uR).float().unsqueeze(-1)
        else:
            pairwise_differences = u.unsqueeze(1) - uR.unsqueeze(0)
        pairwise_distances = calc_pairwise_dist2(u, uR)

        if not self.use_direction_mu_nu:
            if not self.use_x_mu_nu:
                mu_nu_in = yR_encoded
            else:
                mu_nu_in = torch.cat(
                    [yR_encoded, XR.unsqueeze(0).repeat(yR_encoded.size(0), 1, 1)],
                    dim=-1,
                )
        else:
            mu_nu_in = torch.cat(
                [
                    yR_encoded.unsqueeze(1).repeat(
                        1, pairwise_differences.size(0), 1, 1
                    ),
                    pairwise_differences.unsqueeze(0).repeat(
                        yR_encoded.size(0), 1, 1, 1
                    ),
                ],
                dim=3,
            )

        if self.train_separate_extrapolate:
            pGA = GA.unsqueeze(-1) * pairwise_differences
            extrapolating = pGA.sum(1).abs() - pGA.abs().sum(1) == 0
            mu_nu_in = torch.cat(
                [
                    mu_nu_in,
                    extrapolating.float()
                    .unsqueeze(1)
                    .unsqueeze(0)
                    .repeat(mu_nu_in.size(0), 1, mu_nu_in.size(2), 1),
                ],
                dim=3,
            )

        pz_mean_R, pz_logscale_R = torch.split(
            self.mu_nu_theta(mu_nu_in), self.dim_z, -1
        )

        if self.weighted_graph:
            W = norm_graph_weighted(GA, pairwise_distances.add(1e-8).reciprocal())
        else:
            W = self.norm_graph(GA)

        if not self.use_direction_mu_nu:
            pz_mean_all = torch.matmul(W, pz_mean_R)
            pz_logscale_all = torch.matmul(W, pz_logscale_R)
        else:
            pz_mean_all = W.unsqueeze(0).unsqueeze(3).mul(pz_mean_R).sum(2)
            pz_logscale_all = W.unsqueeze(0).unsqueeze(3).mul(pz_logscale_R).sum(2)
        pz_logscale_all = torch.log(
            minscale + (1 - minscale) * F.softplus(pz_logscale_all)
        )
        return pz_mean_all, pz_logscale_all

    def calc_qz_dist(self, X_all, y_all, minscale=1e-8):
        y_all_encoded = self.calc_trans_cond_y(y_all)
        if self.use_x_mu_nu:
            y_all_encoded = torch.cat(
                [y_all_encoded, X_all.unsqueeze(0).repeat(y_all_encoded.size(0), 1, 1)],
                dim=-1,
            )
        if self.use_direction_mu_nu and not self.train_separate_proposal:
            y_all_encoded = torch.cat(
                [
                    y_all_encoded,
                    torch.zeros(
                        *y_all_encoded.shape[0:2], 1, device=y_all_encoded.device
                    ),
                ],
                dim=2,
            )
        qz_mean_all, qz_logscale_all = torch.split(
            self.mu_nu_proposal(y_all_encoded), self.dim_z, -1
        )
        qz_logscale_all = torch.log(
            minscale + (1 - minscale) * F.softplus(qz_logscale_all)
        )
        return qz_mean_all, qz_logscale_all

    def calc_log_pqz(self, pz, qz, z, n_ref):
        """
        Calculates the log difference between pz and qz (with an optional free bits strategy that
        Derek doesn't understand)
        """
        pqz_all = pz.log_prob(z) - qz.log_prob(z)
        assert torch.isnan(pqz_all).sum() == 0
        if self.fb_z > 0:
            log_qpz = -torch.sum(pqz_all)

            if self.training:
                if log_qpz.item() > self.fb_z * z.size(0) * z.size(1) * (1 + 0.05):
                    self.lambda_z = torch.clamp(
                        self.lambda_z * (1 + 0.1), min=1e-8, max=1.0
                    )
                elif log_qpz.item() < self.fb_z * z.size(0) * z.size(1):
                    self.lambda_z = torch.clamp(
                        self.lambda_z * (1 - 0.1), min=1e-8, max=1.0
                    )

            log_pqz_R = self.lambda_z * torch.sum(pqz_all[0:n_ref])
            log_pqz_M = self.lambda_z * torch.sum(pqz_all[n_ref:])

        else:
            log_pqz_R = torch.sum(pqz_all[0:n_ref])
            log_pqz_M = torch.sum(pqz_all[n_ref:])
        return log_pqz_R, log_pqz_M

    def calc_output_y(self, final_rep):
        output_y = self.output(final_rep)
        mean_y, logstd_y = torch.split(output_y, self.dim_y, -1)
        logstd_y = torch.log(0.1 + 0.9 * F.softplus(logstd_y))
        return mean_y, logstd_y

    def calc_py_dist(self, z, u):
        final_rep = (
            z
            if not self.use_plus
            else torch.cat([z, u.unsqueeze(0).repeat(z.size(0), 1, 1)], dim=-1)
        )
        # Encodes y from z
        mean_y, logstd_y = self.calc_output_y(final_rep)
        return mean_y, logstd_y

    def forward(self, XR, yR, XM, yM, G_in=None, A_in=None, kl_anneal=1.0):
        n_ref = XR.size(0)
        X_all = torch.cat([XR, XM], dim=0)
        y_all = torch.cat([yR, yM], dim=1)

        u, uR, uM = self.sample_u(X_all, n_ref)
        assert torch.isnan(u).sum() == 0
        if (A_in is None) or (G_in is None):
            G, A = self.sample_dependency_matrices(uR, uM)
        if G_in is not None:
            G = G_in
        if A_in is not None:
            A = A_in
        GA = torch.cat([G, A], 0)
        # pdb.set_trace()
        assert torch.isnan(GA).sum() == 0

        pz_mean_all, pz_logscale_all = self.calc_pz_dist(u, uR, GA, XR, yR)
        qz_mean_all, qz_logscale_all = self.calc_qz_dist(X_all, y_all)

        assert torch.isnan(pz_mean_all).sum() == 0

        pz = Normal(pz_mean_all, pz_logscale_all)
        qz = Normal(qz_mean_all, qz_logscale_all)
        z = qz.sample()

        log_pqz_R, log_pqz_M = self.calc_log_pqz(pz, qz, z, n_ref)

        mean_y, logstd_y = self.calc_py_dist(z, u)

        mean_yR, mean_yM = torch.split(mean_y, [n_ref, mean_y.size(1) - n_ref], 1)
        logstd_yR, logstd_yM = torch.split(
            logstd_y, [n_ref, logstd_y.size(1) - n_ref], 1
        )

        pyR = Normal(mean_yR, logstd_yR)
        log_pyR = torch.sum(pyR.log_prob(yR))
        assert not torch.isnan(log_pyR)

        pyM = Normal(mean_yM, logstd_yM)
        log_pyM = torch.sum(pyM.log_prob(yM))
        assert not torch.isnan(log_pyM)

        obj_R = (log_pyR + log_pqz_R) / float(XM.size(0))
        obj_M = (log_pyM + log_pqz_M) / float(XM.size(0))

        assert not torch.isnan(obj_R)
        assert not torch.isnan(obj_M)

        if self.condition_on_ref:
            obj = obj_M
        else:
            obj = obj_R + obj_M

        loss = -obj

        return loss

    def inverse_transform(self, y):
        y = y.squeeze(-3)
        y_flat = y.reshape(*y.shape[0:2], -1).cpu().detach().numpy()
        y_flat_invt = self.transf_y.inverse_transform(y_flat)
        y_out = torch.from_numpy(y_flat_invt).resize_(y.size())
        return y_out

    def predict(self, x_new, XR, yR, sample=True, A_in=None, sample_Z=True):
        n_ref = XR.size(0)
        X_all = torch.cat([XR, x_new], 0)
        u, uR, uM = self.sample_u(X_all, n_ref)
        if A_in is None:
            _, A = self.sample_dependency_matrices(uR, uM)
        else:
            A = A_in

        pz_mean_all, pz_logscale_all = self.calc_pz_dist(uM, uR, A, XR, yR)

        pz = Normal(pz_mean_all, pz_logscale_all)
        if sample_Z:
            z = pz.rsample()
        else:
            z = pz_mean_all
        self.z = z
        mean_y, logstd_y = self.calc_py_dist(z, uM)
        py = Normal(mean_y, logstd_y)
        if sample:
            y_new_i = py.sample()
        else:
            y_new_i = mean_y

        if self.transf_y is not None:
            y_pred = self.inverse_transform(y_new_i)
        else:
            y_pred = y_new_i

        return y_pred


class PoolingFNP(RegressionFNP):
    def __init__(
        self,
        pooling_layers=[64],
        pooling_rep_size=32,
        pooling_resid=False,
        set_transformer=False,
        st_numheads=[2, 2],
        **kwargs
    ):
        self.pooling_layers = pooling_layers
        self.pooling_rep_size = pooling_rep_size
        self.set_transformer = set_transformer
        self.st_numheads = st_numheads
        self.pooling_resid = pooling_resid
        super().__init__(**kwargs)

        if not self.set_transformer:
            self.pool_net = self.make_pool_net()
        else:
            self.settrans = self.make_settrans()

    def make_mu_nu_theta(self):
        mu_nu_in = self.dim_y_enc + self.dim_u
        if self.train_separate_extrapolate:
            mu_nu_in += 1
        if self.mu_nu_skip:
            return make_fc_skip(mu_nu_in, self.mu_nu_layers, self.pooling_rep_size)
        else:
            return make_fc_net(mu_nu_in, self.mu_nu_layers, self.pooling_rep_size)

    def make_pool_net(self):
        dim_in = self.mu_nu_theta[-1].out_features
        return make_fc_net(
            dim_in, self.pooling_layers, 2 * self.dim_z, resid=self.pooling_resid
        )

    def make_settrans(self):
        dim_in = self.mu_nu_theta[-1].out_features
        sabs = [SAB(dim_in, dim_in, nh) for nh in self.st_numheads]
        settrans = nn.Sequential(
            *sabs,
            PMA(dim_in, 2, 1, squeeze_out=True),
            make_fc_net(
                dim_in, self.pooling_layers, 2 * self.dim_z, resid=self.pooling_resid
            )
        )
        return settrans

    def calc_pz_dist(self, u, uR, GA, XR, yR, minscale=1e-8):
        yR_encoded = self.calc_trans_cond_y(yR)
        # yR_enc_with_uR = torch.cat([yR_encoded, uR.unsqueeze(0).repeat(yR_encoded.size(0), 1, 1)], dim=-1)

        u_diff = u.unsqueeze(1) - uR.unsqueeze(0)
        mu_nu_in = torch.cat(
            [
                # yR_enc_with_uR.unsqueeze(1).repeat(1, u.size(0), 1, 1),
                yR_encoded.unsqueeze(1).repeat(1, u_diff.size(0), 1, 1),
                u_diff.unsqueeze(0).repeat(yR_encoded.size(0), 1, 1, 1),
            ],
            dim=-1,
        )

        if self.train_separate_extrapolate:
            pGA = GA.unsqueeze(-1) * u_diff
            extrapolating = pGA.sum(1).abs() - pGA.abs().sum(1) == 0
            mu_nu_in = torch.cat(
                [
                    mu_nu_in,
                    extrapolating.float()
                    .unsqueeze(1)
                    .unsqueeze(0)
                    .repeat(mu_nu_in.size(0), 1, mu_nu_in.size(2), 1),
                ],
                dim=3,
            )

        rep_R = self.mu_nu_theta(mu_nu_in)
        if not self.set_transformer:
            rep_pooled = GA.unsqueeze(0).unsqueeze(-1).mul(rep_R).sum(2)
            pz_all = self.pool_net(rep_pooled)
        else:
            pz_all = self.settrans(GA.unsqueeze(0).unsqueeze(-1).mul(rep_R))

        pz_mean_all, pz_logscale_all = torch.split(pz_all, self.dim_z, -1)
        pz_logscale_all = torch.log(
            minscale + (1 - minscale) * F.softplus(pz_logscale_all)
        )
        return pz_mean_all, pz_logscale_all


def make_conv_layer_and_trace(c_in, c_out, kernel_size, stride, dummy_input):
    _, _, h_in, w_in = dummy_input.size()
    q = nn.Conv2d(c_in, c_out, kernel_size, stride)
    dummy_output = q(dummy_input)
    _, _, h_out, w_out = dummy_output.size()
    pad_h = 0 if not ((stride * h_out + kernel_size - 1) == h_in) or stride == 1 else 1
    pad_w = 0 if not ((stride * w_out + kernel_size - 1) == w_in) or stride == 1 else 1
    return q, dummy_output, h_out, w_out, pad_h, pad_w


class ConvRegressionFNP(RegressionFNP):
    """
    Functional Neural Procoess for regression on images
    """

    def __init__(
        self,
        size_h=10,
        size_w=10,
        kernel_sizes=[3, 3],
        strides=[1, 1],
        conv_channels=[20, 20],
        **kwargs
    ):
        self.size_h = size_h
        self.size_w = size_w
        self.kernel_sizes = kernel_sizes
        self.strides = strides
        # self.conv_layers = conv_layers
        self.conv_channels = conv_channels
        super(ConvRegressionFNP, self).__init__(**kwargs)

    def make_trans_cond_y(self):
        # end_channels, self.dim_h_end, self.dim_w_end, self.pad_hs, self.pad_ws = calc_conv2d_layers_output_size(self.conv_layers, 1, self.kernel_size, self.stride, self.size_h, self.size_w)
        # assert self.dim_h_end > 0
        # assert self.dim_w_end > 0
        # self.dim_y_enc = end_channels * self.dim_h_end * self.dim_w_end

        self.pad_hs = []
        self.pad_ws = []

        dummy_input = torch.randn(1, 1, self.size_h, self.size_w)

        q, dummy_input, h_out, w_out, pad_h, pad_w = make_conv_layer_and_trace(
            1, self.conv_channels[0], self.kernel_sizes[0], self.strides[0], dummy_input
        )
        y_encoder_array = [q, nn.ReLU()]
        self.pad_hs.append(pad_h)
        self.pad_ws.append(pad_w)
        for i in range(len(self.conv_channels) - 1):
            q, dummy_input, h_out, w_out, pad_h, pad_w = make_conv_layer_and_trace(
                self.conv_channels[i],
                self.conv_channels[i + 1],
                self.kernel_sizes[i + 1],
                self.strides[i + 1],
                dummy_input,
            )
            y_encoder_array += [q, nn.ReLU()]
            self.pad_hs.append(pad_h)
            self.pad_ws.append(pad_w)

        self.dim_y_enc = self.conv_channels[-1] * h_out * w_out
        self.dim_h_end = h_out
        self.dim_w_end = w_out
        y_encoder_array.append(Flatten())
        y_encoder_array.append(nn.Linear(self.dim_y_enc, self.dim_y_enc))
        q = nn.Sequential(*y_encoder_array)
        return q

    def calc_trans_cond_y(self, yR):
        yR_encoded = self.trans_cond_y(
            yR.view(yR.size(0) * yR.size(1), yR.size(2), yR.size(3), yR.size(4))
        )
        yR_encoded = yR_encoded.view(yR.size(0), yR.size(1), -1)
        return yR_encoded

    def make_output(self):
        fc_layers = super(ConvRegressionFNP, self).make_output()[:-1]

        output_array = [
            fc_layers,
            nn.Linear(fc_layers[-1][0].out_features, self.dim_y_enc),
            UnFlatten([self.conv_channels[-1], self.dim_h_end, self.dim_w_end]),
        ]

        for i in range(len(self.conv_channels) - 1):
            inchannel = self.conv_channels[-(i + 1)]
            ouchannel = self.conv_channels[-(i + 2)]
            output_array.append(nn.ReLU())
            output_array.append(
                nn.ConvTranspose2d(
                    inchannel,
                    ouchannel,
                    self.kernel_sizes[-(i + 1)],
                    self.strides[-(i + 1)],
                    output_padding=(self.pad_hs[-(i + 1)], self.pad_ws[-(i + 1)]),
                )
            )

        output_array += [
            nn.ReLU(),
            nn.ConvTranspose2d(
                self.conv_channels[0], 2, self.kernel_sizes[0], self.strides[0]
            ),
        ]

        return nn.Sequential(*output_array)

    def calc_output_y(self, final_rep):
        output_y = self.output(
            final_rep.view(final_rep.size(0) * final_rep.size(1), final_rep.size(2))
        )
        output_y = output_y.view(
            final_rep.size(0),
            final_rep.size(1),
            output_y.size(1),
            output_y.size(2),
            output_y.size(3),
        )
        mean_y = output_y[:, :, 0:1, :, :]
        logstd_y = output_y[:, :, 1:2, :, :]
        logstd_y = torch.log(0.1 + 0.9 * F.softplus(logstd_y))
        return mean_y, logstd_y


class ConvPoolingFNP(PoolingFNP, ConvRegressionFNP):
    def __init__(self, **kwargs):
        super(ConvPoolingFNP, self).__init__(**kwargs)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
float_tensor = (
    torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
)


def norm_graph_weighted(x, weights):
    xw = torch.mul(x, weights)
    return xw / (xw.sum(dim=1, keepdim=True) + 1e-8)


def logitexp(logp):
    # https: // github.com / pytorch / pytorch / issues / 4007
    pos = torch.clamp(logp, min=-0.69314718056)
    neg = torch.clamp(logp, max=-0.69314718056)
    neg_val = neg - torch.log(1 - torch.exp(neg))
    pos_val = -torch.log(torch.clamp(torch.expm1(-pos), min=1e-20))
    return pos_val + neg_val


def one_hot(x, n_classes=10):
    x_onehot = float_tensor(x.size(0), n_classes).zero_()
    x_onehot.scatter_(1, x[:, None], 1)

    return x_onehot


class LogitRelaxedBernoulli(object):
    def __init__(self, logits, temperature=0.3, **kwargs):
        self.logits = logits
        self.temperature = temperature

    def rsample(self):
        eps = torch.clamp(
            torch.rand(
                self.logits.size(), dtype=self.logits.dtype, device=self.logits.device
            ),
            min=1e-6,
            max=1 - 1e-6,
        )
        y = (self.logits + torch.log(eps) - torch.log(1.0 - eps)) / self.temperature
        return y

    def log_prob(self, value):
        return (
            math.log(self.temperature)
            - self.temperature * value
            + self.logits
            - 2 * F.softplus(-self.temperature * value + self.logits)
        )


class Normal(object):
    def __init__(self, means, logscales, **kwargs):
        self.means = means
        self.logscales = logscales

    def log_prob(self, value):
        log_prob = torch.pow(value - self.means, 2)
        log_prob *= -(1 / (2.0 * self.logscales.mul(2.0).exp()))
        log_prob -= self.logscales + 0.5 * math.log(2.0 * math.pi)
        return log_prob

    def sample(self, **kwargs):
        eps = torch.normal(
            float_tensor(self.means.size()).zero_(),
            float_tensor(self.means.size()).fill_(1),
        )
        return self.means + self.logscales.exp() * eps

    def rsample(self, **kwargs):
        return self.sample(**kwargs)


class L1Error(object):
    def __init__(self, means, logscales, **kwargs):
        self.means = means
        self.logscales = logscales

    def log_prob(self, value):
        log_prob = torch.abs(value - self.means)
        log_prob *= -(1 / (2.0 * self.logscales.mul(1.0).exp()))
        log_prob -= self.logscales + 0.5 * math.log(2.0 * math.pi)
        return log_prob

    def sample(self, **kwargs):
        eps = torch.distributions.Laplace(
            float_tensor(self.means.size()).zero_(),
            float_tensor(self.means.size()).fill_(1),
        ).sample()
        return self.means + self.logscales.exp() * eps

    def rsample(self, **kwargs):
        return self.sample(**kwargs)


def order_z(z):
    # scalar ordering function
    if z.size(1) == 1:
        return z
    log_cdf = torch.sum(
        torch.log(0.5 + 0.5 * torch.erf(z / math.sqrt(2))), dim=1, keepdim=True
    )
    return log_cdf


def sample_DAG(Z, g, training=True, temperature=0.3):
    # get the indices of an upper triangular adjacency matrix that represents the DAG
    idx_utr = np.triu_indices(Z.size(0), 1)

    # get the ordering
    ordering = order_z(Z)
    # sort the latents according to the ordering
    sort_idx = torch.sort(torch.squeeze(ordering), 0)[1]
    Y = Z[sort_idx, :]
    # form the latent pairs for the edges
    Z_pairs = torch.cat([Y[idx_utr[0]], Y[idx_utr[1]]], 1)
    # get the logits for the edges in the DAG
    logits = g(Z_pairs)

    if training:
        p_edges = LogitRelaxedBernoulli(logits=logits, temperature=temperature)
        G = torch.sigmoid(p_edges.rsample())
    else:
        p_edges = Bernoulli(logits=logits)
        G = p_edges.sample()

    # embed the upper triangular to the adjacency matrix
    unsorted_G = float_tensor(Z.size(0), Z.size(0)).zero_()
    unsorted_G[idx_utr[0], idx_utr[1]] = G.squeeze()
    # unsort the dag to conform to the data order
    original_idx = torch.sort(sort_idx)[1]
    unsorted_G = unsorted_G[original_idx, :][:, original_idx]

    return unsorted_G


def sample_bipartite(Z1, Z2, g, training=True, temperature=0.3):
    indices = []
    for element in product(range(Z1.size(0)), range(Z2.size(0))):
        indices.append(element)
    indices = np.array(indices)
    Z_pairs = torch.cat([Z1[indices[:, 0]], Z2[indices[:, 1]]], 1)

    logits = g(Z_pairs)
    if training:
        p_edges = LogitRelaxedBernoulli(logits=logits, temperature=temperature)
        A_vals = torch.sigmoid(p_edges.rsample())
    else:
        p_edges = Bernoulli(logits=logits)
        A_vals = p_edges.sample()

    # embed the values to the adjacency matrix
    A = float_tensor(Z1.size(0), Z2.size(0)).zero_()
    A[indices[:, 0], indices[:, 1]] = A_vals.squeeze()

    return A


class Flatten(torch.nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        assert len(x.shape) > 1

        return x.view(x.shape[0], -1)


class UnFlatten(torch.nn.Module):
    def __init__(self, shape):
        super(UnFlatten, self).__init__()
        self.shape = shape

    def forward(self, x):
        # assert len(x.shape) == 2
        return x.view(*x.shape[0:-1], *self.shape)


class ResidualLayer(torch.nn.Module):
    def __init__(self, f):
        super(ResidualLayer, self).__init__()
        self.f = f

    def forward(self, x):
        y = self.f(x)
        if y.size(-1) > x.size(-1):
            zshape = [*y.size()]
            zshape[-1] = y.size(-1) - x.size(-1)
            xx = torch.cat([x, torch.zeros(zshape, device=x.device)], dim=-1)
        else:
            xx = x
        return y + xx


## Set Transformer
class MAB(nn.Module):
    def __init__(self, dim_Q, dim_K, dim_V, num_heads, ln=False):
        super(MAB, self).__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_K, dim_V)
        self.fc_v = nn.Linear(dim_K, dim_V)
        if ln:
            self.ln0 = nn.LayerNorm(dim_V)
            self.ln1 = nn.LayerNorm(dim_V)
        self.fc_o = nn.Linear(dim_V, dim_V)

    def forward(self, Q, K):
        # We assume that Q and K are ...x N x D, where
        # ... are 0 or more preceding dimensions.
        Q = self.fc_q(Q).unsqueeze(-3)
        K, V = self.fc_k(K).unsqueeze(-3), self.fc_v(K).unsqueeze(-3)

        dim_split = self.dim_V // self.num_heads
        Q_ = torch.cat(Q.split(dim_split, -1), -3)
        K_ = torch.cat(K.split(dim_split, -1), -3)
        V_ = torch.cat(V.split(dim_split, -1), -3)

        A = torch.softmax(Q_.matmul(K_.transpose(-2, -1)) / math.sqrt(self.dim_V), -1)
        O = torch.cat((Q_ + A.matmul(V_)).split(Q.size(-3), -3), -1)
        O = O if getattr(self, "ln0", None) is None else self.ln0(O)
        O = O + F.relu(self.fc_o(O))
        O = O if getattr(self, "ln1", None) is None else self.ln1(O)
        return O.squeeze(-3)


class SAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, ln=False):
        super(SAB, self).__init__()
        self.mab = MAB(dim_in, dim_in, dim_out, num_heads, ln=ln)

    def forward(self, X):
        return self.mab(X, X)


class ISAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, num_inds, ln=False):
        super(ISAB, self).__init__()
        self.I = nn.Parameter(torch.Tensor(1, num_inds, dim_out))
        nn.init.xavier_uniform_(self.I)
        self.mab0 = MAB(dim_out, dim_in, dim_out, num_heads, ln=ln)
        self.mab1 = MAB(dim_in, dim_out, dim_out, num_heads, ln=ln)

    def forward(self, X):
        H = self.mab0(self.I.repeat(X.size(0), 1, 1), X)
        return self.mab1(X, H)


class PMA(nn.Module):
    def __init__(self, dim, num_heads, num_seeds, ln=False, squeeze_out=False):
        super(PMA, self).__init__()
        self.S = nn.Parameter(torch.Tensor(num_seeds, dim))
        nn.init.xavier_uniform_(self.S)
        self.mab = MAB(dim, dim, dim, num_heads, ln=ln)
        self.squeeze_out = squeeze_out

    def forward(self, X):
        diff_dim = len(X.size()) - 2
        S = self.S[(None,) * diff_dim].expand(*X.shape[:-2], *self.S.shape)
        out = self.mab(S, X)
        if self.squeeze_out:
            out = out.squeeze(-2)
        return out


## 1-dimensional example
