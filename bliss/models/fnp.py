import torch.nn as nn
import torch
import math
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np
import matplotlib.pyplot as plt
from torch.distributions import Bernoulli, MultivariateNormal
from itertools import product
from scipy.signal import savgol_filter

# from .utils import Normal, L1Error, float_tensor, logitexp, sample_DAG, sample_bipartite, Flatten, UnFlatten, one_hot, norm_graph_weighted, ResidualLayer
# from .set_transformer.modules import SAB, PMA
from torch.distributions import Categorical, Bernoulli
from torch.optim import Adam

from sklearn.preprocessing import StandardScaler


class MLP(nn.Module):
    def __init__(self, in_features, hs, out_features, act=nn.ReLU, final=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        layers = [nn.Sequential(nn.Linear(in_features, hs[0]), act())]
        for i in range(len(hs) - 1):
            layers.append(nn.Sequential(nn.Linear(hs[i], hs[i + 1]), act()))
        layers.append(nn.Linear(hs[-1], out_features))
        if final is not None:
            layers += [final()]
        self.f = nn.Sequential(*layers)

    def forward(self, X):
        return self.f(X)


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


# class FNP(pl.LightningModule):
#     """
#     This is an implementation of the Functional Neural Process (FNP)
#     from http://arxiv.org/abs/1906.08324.

#     The FNP is made up of a few different modules. These are all
#     application-specific, so they are passed as arguments to the
#     class constructor
#     * cov_vencoder This is a module which, conditional on
#     the covariate X, samples a representation U (probabilistic)
#     * rep_encoder: This is a module which deterministically
#     encodes the covariate representation U and the label Y
#     * prop_vencoder: This is a module which samples a representation
#     for an object given the label (only used in training, not prediction)
#     * rep_vpooler: This is a module which samples a representation
#     based on pooling the representations of its dependencies.
#     * label_vdecoder: This is a module which probabilistically
#     decodes the pooled representation into the output

#     Part of the FNP is also learning and sampling from a
#     dependency graph. This can be modified via an optional argument:
#     * dep_graph_sampler
#     """
#     def __init__(self, cov_vencoder, rep_encoder, prop_encoder, rep_vpooler, label_vdecoder, dep_graph_sampler=BernoulliGraph):
#        super().__init__(self)

#     def forward(self):
#         """
#         This step does everything short of actually sampling the labels
#         """

#     def log_prob(self):
#         """
#         This step calculates the log_probability over the current dataset
#         """
#         self.forward()
#     def predict(self):
#         """
#         This step predicts labels for the dependent objects
#         """
#         self.forward()

#     ## Lightning functions for training
#     def training_step(self, batch, batch_idx):
#         pass

#     def validation_step(self, batch, batch_idx):
#         pass

#     def configure_optimizers(self):
#         pass
class VEncoder(nn.Module):
    def __init__(self, dim_x, dim_h, dim_u, n_layers):
        super().__init__()
        self.dim_u = dim_u
        self.encoder = MLP(dim_x, [dim_h] * n_layers, 2 * dim_u)

    def forward(self, X_all):
        mean_z, logscale_z = torch.split(self.encoder(X_all), self.dim_u, -1)
        pz = Normal(mean_z, logscale_z)
        z = pz.rsample()
        return z


class AveragePooler(nn.Module):
    def __init__(
        self,
        dim_z,
    ):
        super().__init__()
        self.dim_z = dim_z

        # normalizes the graph such that inner products correspond to averages of the parents
        self.norm_graph = lambda x: x / (torch.sum(x, 1, keepdim=True) + 1e-8)

    def calc_pz_dist(self, rep_R, GA, minscale=1e-8):
        pz_mean_R, pz_logscale_R = torch.split(rep_R, self.dim_z, -1)

        W = self.norm_graph(GA)

        pz_mean_all = torch.matmul(W, pz_mean_R)
        pz_logscale_all = torch.matmul(W, pz_logscale_R)
        pz_logscale_all = torch.log(
            minscale + (1 - minscale) * F.softplus(pz_logscale_all)
        )
        return pz_mean_all, pz_logscale_all


class SetPooler(nn.Module):
    def __init__(
        self,
        dim_rep_R,
        dim_z,
        pooling_layers,
        set_transformer,
        st_numheads,
    ):
        super().__init__()
        self.dim_rep_R = dim_rep_R
        self.dim_z = dim_z
        self.pooling_layers = pooling_layers
        self.set_transformer = set_transformer
        self.st_numheads = st_numheads

        # normalizes the graph such that inner products correspond to averages of the parents
        self.norm_graph = lambda x: x / (torch.sum(x, 1, keepdim=True) + 1e-8)

        if not self.set_transformer:
            self.pool_net = MLP(self.dim_rep_R, self.pooling_layers, 2 * self.dim_z)
            self.settrans = None
        else:
            self.pool_net = None
            dim_in = self.dim_rep_R
            sabs = [SAB(dim_in, dim_in, nh) for nh in self.st_numheads]
            self.settrans = nn.Sequential(
                *sabs,
                PMA(dim_in, 2, 1, squeeze_out=True),
                MLP(dim_in, self.pooling_layers, 2 * self.dim_z)
            )

    def calc_pz_dist(self, rep_R, GA, minscale=1e-8):
        # yR_enc_with_uR = torch.cat([yR_encoded, uR.unsqueeze(0).repeat(yR_encoded.size(0), 1, 1)], dim=-1)

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


class RepEncoder(nn.Module):
    def __init__(self, mu_nu_theta, use_u_diff=False, use_x=False):
        super().__init__()
        self.mu_nu_theta = mu_nu_theta
        self.use_u_diff = use_u_diff
        self.use_x = use_x

    def forward(self, u, uR, XR, yR_encoded):
        input_list = [yR_encoded]
        if self.use_x:
            input_list.append(XR.unsqueeze(0).repeat(input_list[0].size(0), 1, 1))

        ## If we look at differences in U values, we need to increase the dimension
        ## (each representative member looks different to each dependent member)
        if self.use_u_diff:
            u_diff = u.unsqueeze(1) - uR.unsqueeze(0)
            for i, x in enumerate(input_list):
                input_list[i] = x.unsqueeze(1).repeat(1, u_diff.size(0), 1, 1)
            input_list.append(u_diff.unsqueeze(0).repeat(x.size(0), 1, 1, 1))

        mu_nu_in = torch.cat(input_list, -1)
        rep_R = self.mu_nu_theta(mu_nu_in)
        return rep_R


class SplitLayer(nn.Module):
    """
    This layer splits the input according to the arguments to torch.split
    """

    def __init__(self, split_size_or_sections, dim):
        super().__init__()
        self.split_size_or_sections = split_size_or_sections
        self.dim = dim

    def forward(self, tensor):
        return torch.split(tensor, self.split_size_or_sections, self.dim)


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
        output_layers=[128],
        x_as_u=False,
        condition_on_ref=False,
        discrete_orientation=True,
        weighted_graph=False,
        pooler=None,
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
        self.output_layers = output_layers
        self.x_as_u = x_as_u
        self.condition_on_ref = condition_on_ref
        self.discrete_orientation = discrete_orientation
        self.weighted_graph = weighted_graph

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
        if not self.x_as_u:
            self.cov_vencoder = VEncoder(dim_x, self.dim_h, self.dim_u, n_layers)
        else:
            self.cov_vencoder = nn.Identity()
        # p(u|x)
        # q(z|x)
        # See equation 7
        self.q_z = nn.Linear(self.dim_h, 2 * self.dim_z)
        # for p(z|A, XR, yR)

        self.dim_y_enc = 2 * self.dim_z
        self.trans_cond_y = MLP(
            self.dim_y,
            self.y_encoder_layers,
            2 * self.dim_z,
        )
        mu_nu_in = self.dim_y_enc
        if self.use_x_mu_nu is True:
            mu_nu_in += self.dim_x
        if self.use_direction_mu_nu:
            mu_nu_in += 1
        mu_nu_theta = MLP(mu_nu_in, self.mu_nu_layers, 2 * self.dim_z)
        self.rep_encoder = RepEncoder(mu_nu_theta, use_u_diff=False, use_x=use_x_mu_nu)
        if pooler is None:
            self.pooler = AveragePooler(
                dim_z,
            )
        else:
            self.pooler = pooler

        self.mu_nu_proposal = self.make_mu_nu_proposal()
        # for p(y|z)
        output_insize = self.dim_z if not self.use_plus else self.dim_z + self.dim_u
        self.output = nn.Sequential(
            MLP(output_insize, self.output_layers, 2 * self.dim_y),
            SplitLayer(self.dim_y, -1),
        )

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

    def make_mu_nu_proposal(self):
        mu_nu_in = self.dim_y_enc
        if self.use_x_mu_nu is True:
            # raise(NotImplementedError("use_x_mu_nu is not yet implemented")
            mu_nu_in += self.dim_x
        return MLP(mu_nu_in, self.mu_nu_layers, 2 * self.dim_z)

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

    def calc_qz_dist(self, X_all, y_all, minscale=1e-8):
        y_all_encoded = self.trans_cond_y(y_all)
        if self.use_x_mu_nu:
            y_all_encoded = torch.cat(
                [y_all_encoded, X_all.unsqueeze(0).repeat(y_all_encoded.size(0), 1, 1)],
                dim=-1,
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
        mean_y, logstd_y = self.output(final_rep)
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

        u = self.cov_vencoder(X_all)
        uR = u[:n_ref]
        uM = u[n_ref:]

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

        yR_encoded = self.trans_cond_y(yR)
        rep_R = self.rep_encoder(u, uR, XR, yR_encoded)
        pz_mean_all, pz_logscale_all = self.pooler.calc_pz_dist(rep_R, GA)
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
        u = self.cov_vencoder(X_all)
        uR = u[:n_ref]
        uM = u[n_ref:]
        if A_in is None:
            _, A = self.sample_dependency_matrices(uR, uM)
        else:
            A = A_in

        yR_encoded = self.trans_cond_y(yR)
        rep_R = self.rep_encoder(uM, uR, XR, yR_encoded)
        pz_mean_all, pz_logscale_all = self.pooler.calc_pz_dist(rep_R, A)

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
        set_transformer=False,
        st_numheads=[2, 2],
        **kwargs
    ):
        self.pooling_layers = pooling_layers
        self.pooling_rep_size = pooling_rep_size
        self.set_transformer = set_transformer
        self.st_numheads = st_numheads
        super().__init__(**kwargs)
        mu_nu_in = self.dim_y_enc + self.dim_u
        mu_nu_theta = MLP(mu_nu_in, self.mu_nu_layers, self.pooling_rep_size)
        self.rep_encoder = RepEncoder(mu_nu_theta, use_u_diff=True, use_x=False)

        self.pooler = SetPooler(
            mu_nu_theta.out_features,
            self.dim_z,
            self.pooling_layers,
            self.set_transformer,
            self.st_numheads,
        )


def make_conv_layer_and_trace(c_in, c_out, kernel_size, stride, dummy_input):
    _, _, h_in, w_in = dummy_input.size()
    q = nn.Conv2d(c_in, c_out, kernel_size, stride)
    dummy_output = q(dummy_input)
    _, _, h_out, w_out = dummy_output.size()
    pad_h = 0 if not ((stride * h_out + kernel_size - 1) == h_in) or stride == 1 else 1
    pad_w = 0 if not ((stride * w_out + kernel_size - 1) == w_in) or stride == 1 else 1
    return q, dummy_output, h_out, w_out, pad_h, pad_w


class ReshapeWrapper(nn.Module):
    """
    This module wraps around a module which expects tensors of a fixed dimension. For example,
    a Conv2D layer expects a 4-dimensional tensor, but we may want to use 5-dimensional or higher
    tensors with it. This flattens the higher dimensions into dimension 0, then unflattens them.
    """

    def __init__(self, f, f_dim):
        super().__init__()
        self.f = f
        self.f_dim = f_dim

    def forward(self, X):
        k = len(X.shape) - self.f_dim + 1
        assert k > 0
        in_size = torch.Size([np.product(X.shape[:k])]) + X.shape[k:]
        Y = self.f(X.view(in_size))
        Y = Y.view(X.shape[:k] + Y.shape[(k - 1) :])
        return Y


class Conv2DAutoEncoder(nn.Module):
    """
    This module creates a stacked layer of Conv2D layers to decode an image tensor to a
    flattened representation. It simulatenously creates a corresponding stacked layer of
    Conv2d Transposes which will map that representation to an output image of the same
    dimension.
    """

    def __init__(
        self,
        size_h,
        size_w,
        conv_channels,
        kernel_sizes,
        strides,
        output_insize,
        output_layers,
    ):
        super().__init__()
        self.size_h = size_h
        self.size_w = size_w
        self.conv_channels = conv_channels
        self.kernel_sizes = kernel_sizes
        self.strides = strides
        self.output_insize = output_insize
        self.output_layers = output_layers

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
        self.encoder = ReshapeWrapper(nn.Sequential(*y_encoder_array), 4)

        ## Make Convolutional Output
        fc_layer = MLP(self.output_insize, self.output_layers, self.dim_y_enc)
        output_array = []
        for i in range(len(self.conv_channels) - 1):
            inchannel = self.conv_channels[-(i + 1)]
            ouchannel = self.conv_channels[-(i + 2)]
            output_array.append(
                nn.ConvTranspose2d(
                    inchannel,
                    ouchannel,
                    self.kernel_sizes[-(i + 1)],
                    self.strides[-(i + 1)],
                    output_padding=(self.pad_hs[-(i + 1)], self.pad_ws[-(i + 1)]),
                )
            )
            output_array.append(nn.ReLU())

        output_array += [
            nn.ConvTranspose2d(
                self.conv_channels[0], 2, self.kernel_sizes[0], self.strides[0]
            ),
        ]

        self.decoder = nn.Sequential(
            fc_layer,
            UnFlatten([self.conv_channels[-1], self.dim_h_end, self.dim_w_end]),
            ReshapeWrapper(nn.Sequential(*output_array), 4),
            SplitLayer(1, -3),
        )


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

        conv_autoencoder = Conv2DAutoEncoder(
            size_h,
            size_w,
            conv_channels,
            kernel_sizes,
            strides,
            self.dim_z if not self.use_plus else self.dim_z + self.dim_u,
            self.output_layers,
        )

        self.trans_cond_y = conv_autoencoder.encoder
        self.dim_y_enc = conv_autoencoder.dim_y_enc
        self.mu_nu_proposal = self.make_mu_nu_proposal()
        self.output = conv_autoencoder.decoder


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


## ----------------------------
## 1-dimensional example
## ----------------------------
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


def train_onedim_model(model, od, epochs=10000, lr=1e-4, visualize=False):
    if torch.cuda.is_available():
        model = model.cuda()
    optimizer = Adam(model.parameters(), lr=lr)
    model.train()
    holdout_loss_prev = np.infty
    holdout_loss_initial = model(od.XR, od.yhR, od.XM, od.yhM)[0]
    holdout_loss_best = holdout_loss_initial
    print("Initial holdout loss: {:.3f})".format(holdout_loss_initial))
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
            holdout_loss = model(od.XR, od.yhR, od.XM, od.yhM)[0]
            if holdout_loss < holdout_loss_best:
                holdout_loss_best = holdout_loss
            print("Holdout loss: {:.3f}".format(holdout_loss))
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
    print("Done.")
    return model, holdout_loss_initial, holdout_loss, holdout_loss_best


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


def make_rot_matrices(phis):
    cos_phi = phis.cos()
    sin_phi = phis.sin()
    row1 = torch.stack([cos_phi, -sin_phi], -1)
    row2 = torch.stack([sin_phi, cos_phi], -1)
    y = torch.stack([row1, row2], -1)
    return y


def make_X_ref_pair(x1, x2, K):
    X = x1 + (x2 - x1) / (K + 1) * (torch.tensor(range(K), dtype=torch.float32) + 1.0)
    return X


# TODO: Add jitter here
def make_X(n_ref, Ks):
    X_ref = torch.tensor(range(0, n_ref * 2, 2), dtype=torch.float32)
    X_dep = torch.tensor([], dtype=torch.float32)
    for i in range(n_ref - 1):
        X_dep = torch.cat([X_dep, make_X_ref_pair(X_ref[i], X_ref[i + 1], Ks[i])])
    X_all, idxs = torch.cat([X_ref, X_dep], 0).sort()

    Is = torch.tensor(range(X_all.size(0)))
    idx_ref = Is[idxs < n_ref]
    idx_dep = Is[idxs >= n_ref]
    return X_ref, X_dep, X_all, idx_ref, idx_dep


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


def make_graphs(n_ref, Ks):
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
        A = torch.cat([A, make_graph_ref_pair(j, j + 1, Ks[j], n_ref)])

    return G, A


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
        rots = make_rot_matrices(phi)
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

        self.X_ref, self.X_dep, self.X_all, self.idx_ref, self.idx_dep = make_X(
            n_ref, Ks
        )
        self.G, self.A = make_graphs(n_ref, Ks)

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

    def cpu(self):
        self.X_r, self.X_m, self.X = self.X_r.cpu(), self.X_m.cpu(), self.X.cpu()
        self.y_r, self.y_m, self.y = self.y_r.cpu(), self.y_m.cpu(), self.y.cpu()

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
