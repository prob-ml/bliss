import torch.nn as nn
import torch
import math
import torch.nn.functional as F
import numpy as np
from torch.distributions import Bernoulli
from torch.distributions.relaxed_bernoulli import LogitRelaxedBernoulli
from itertools import product

# from .utils import Normal, L1Error, float_tensor, logitexp, sample_DAG, sample_bipartite, Flatten, UnFlatten, one_hot, norm_graph_weighted, ResidualLayer
# from .set_transformer.modules import SAB, PMA
from torch.distributions import Normal, Bernoulli


class FNP(nn.Module):
    """
    This is an implementation of the Functional Neural Process (FNP)
    from http://arxiv.org/abs/1906.08324.

    The FNP is made up of a few different modules. These are all
    application-specific, so they are passed as arguments to the
    class constructor
    * cov_vencoder This is a module which takes input tensor X
    and outputs a probability distribution from which U is sampled.
    * dep_graph: An object of type DepGraph which samples graphs G and A.
    Requires methods .sample_G(uR) and .sample_A(uM, uR)
    * trans_cond_y: A module which takes labels Y and returns a flattened
    representation Y_encoded
    * rep_encoder: This is a module which takes (u, uR, XR, yR_encoded) as input
    and outputs a flattened representation. (see RepEncoder)
    * pooler: This is a module which takes the output of rep_encoder and the dependency matrix
    and returns a sampler for the latent variables Z. (see AveragePooler, SetPooler)
    * prop_vencoder: This is a module which samples a representation
    for an object given X and yR_encoded (only used in training, not prediction)
    * label_vdecoder: This is a module which probabilistically
    decodes the pooled representation into the output
    * fb_z: If non-zero, the amount of "free-bits" regularization to apply while training. This
    encourages learning a better representation and can avoid a local minimum in the prior.
    """

    def __init__(
        self,
        cov_vencoder,
        dep_graph,
        trans_cond_y,
        rep_encoder,
        pooler,
        prop_vencoder,
        label_vdecoder,
        fb_z=0.0,
    ):
        super().__init__()
        ## Learned Submodules
        self.cov_vencoder = cov_vencoder
        self.dep_graph = dep_graph
        self.trans_cond_y = trans_cond_y
        self.rep_encoder = rep_encoder
        self.pooler = pooler
        self.prop_vencoder = prop_vencoder
        self.label_vdecoder = label_vdecoder

        ## Initialize free-bits regularization
        self.fb_z = fb_z
        self.register_buffer("lambda_z", torch.tensor(1e-8))

    def encode(self, XR, yR, XM, G_in=None, A_in=None, mode="infer"):
        """
        This method runs the FNP up to the point the distributions for the latent
        variables are calculated. This is the shared procedure for both model inference
        and prediction.
        """
        n_ref = XR.size(0)
        X_all = torch.cat([XR, XM], dim=0)

        ## Sample covariate representation U
        pu = self.cov_vencoder(X_all)
        u = pu.rsample()
        uR = u[:n_ref]
        uM = u[n_ref:]
        assert torch.isnan(u).sum() == 0

        ## Sample the dependency matrices
        ## If we are training ("infer"), the entire dependency
        ## graph (A and G) is generated.
        ## If we are predicting ("predict"), only the dependent
        ## graph (A) is used.
        if A_in is None:
            A = self.dep_graph.sample_A(uM, uR)
        else:
            A = A_in
        if mode == "infer":
            if G_in is None:
                G = self.dep_graph.sample_G(uR)
            else:
                G = G_in

            GA = torch.cat([G, A], 0)
        elif mode == "predict":
            GA = A
        else:
            raise ("invalid encode mode")
        assert torch.isnan(GA).sum() == 0

        ## From the dependency graph GA and the encoded
        ## representative set, we calculate the distribution
        ## of the latent representations Z
        yR_encoded = self.trans_cond_y(yR)
        rep_R = self.rep_encoder(u, uR, XR, yR_encoded)
        pz = self.pooler(rep_R, GA)
        return u, pz

    def log_prob(self, XR, yR, XM, yM, G_in=None, A_in=None):
        ## Get the distribution of the latent representations Z
        ## and the encoding U of the covariates
        u, pz = self.encode(XR, yR, XM, G_in, A_in)

        n_ref = XR.size(0)
        X_all = torch.cat([XR, XM], dim=0)

        ## Sample Z from the proposal distribution (which
        ## is allowed to look at the labels of all points)
        y_all = torch.cat([yR, yM], dim=1)
        y_all_encoded = self.trans_cond_y(y_all)
        qz = self.prop_vencoder(X_all.unsqueeze(0), y_all_encoded)
        z = qz.rsample()

        ## Calculate the difference between the "prior" pz and the
        ## variational distribution qz with an optional "free-bits" strategy.
        ## This free-bits strategy is a lower bound that solves the problem
        ## of posterior collapse.
        log_pqz = self.calc_log_pqz(pz, qz, z)

        ## Calculate the conditional likelihood of the labels y conditional on Z
        py = self.label_vdecoder(z, u.unsqueeze(0))
        log_py = py.log_prob(y_all).sum() / XM.size(0)
        assert not torch.isnan(log_py)
        obj = log_pqz + log_py
        return obj

    def calc_log_pqz(self, pz, qz, z):
        """
        Calculates the log difference between pz and qz (with an optional free bits strategy)
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

            log_pqz = self.lambda_z * pqz_all.sum()

        else:
            log_pqz = pqz_all.sum()
        return log_pqz

    def forward(self, XR, yR, XM, yM, G_in=None, A_in=None, kl_anneal=1.0):
        return -self.log_prob(XR, yR, XM, yM, G_in, A_in)

    def predict(self, x_new, XR, yR, sample=True, A_in=None, sample_Z=True):
        n_ref = XR.size(0)
        u, pz = self.encode(XR, yR, x_new, None, A_in)
        uM = u[n_ref:]
        if sample_Z:
            z = pz.rsample()
        else:
            z = pz.means
        zM = z[:, n_ref:]
        self.z = z

        py = self.label_vdecoder(zM, uM.unsqueeze(0))
        if sample:
            y_pred = py.sample()
        else:
            y_pred = py.means

        return y_pred


## ***********************
## FNP-specific submodules
## ***********************
class DepGraph(nn.Module):
    """
    A dependency-graph module for use within FNP. For tensors of input encodings from
    the reference points (uR) and dependent points (uM), this module returns matrices
    that represent the dependency structure.
    """

    def __init__(self, dim_u, temperature=0.3):
        super().__init__()
        ## Temperature for LogitRelaxedBernoulli when training
        self.temperature = temperature
        ## Initialized the distance function pairwise_g
        ## This has a single learned parameter
        self.g_logscale = nn.Parameter(torch.tensor(math.log(dim_u) * 0.5))
        self.g = lambda x: self.logitexp(
            -0.5
            * torch.sum(torch.pow(x[:, dim_u:] - x[:, 0:dim_u], 2), 1, keepdim=True)
            / self.g_logscale.exp()
        ).view(x.size(0), 1)

    def sample_G(self, uR):
        # get the indices of an upper triangular adjacency matrix that represents the DAG
        idx_utr = np.triu_indices(uR.size(0), 1)

        # get the ordering
        ordering = self.order_z(uR)
        # sort the latents according to the ordering
        sort_idx = torch.sort(torch.squeeze(ordering), 0)[1]
        Y = uR[sort_idx, :]
        # form the latent pairs for the edges
        Z_pairs = torch.cat([Y[idx_utr[0]], Y[idx_utr[1]]], 1)
        # get the logits for the edges in the DAG
        logits = self.g(Z_pairs)

        if self.training:
            p_edges = LogitRelaxedBernoulli(temperature=self.temperature, logits=logits)
            G = torch.sigmoid(p_edges.rsample())
        else:
            p_edges = Bernoulli(logits=logits)
            G = p_edges.sample()

        # embed the upper triangular to the adjacency matrix
        unsorted_G = torch.zeros(uR.size(0), uR.size(0), device=uR.device)
        unsorted_G[idx_utr[0], idx_utr[1]] = G.squeeze()
        # unsort the dag to conform to the data order
        original_idx = torch.sort(sort_idx)[1]
        unsorted_G = unsorted_G[original_idx, :][:, original_idx]

        return unsorted_G

    def sample_A(self, uM, uR):
        indices = []
        for element in product(range(uM.size(0)), range(uR.size(0))):
            indices.append(element)
        indices = np.array(indices)
        Z_pairs = torch.cat([uM[indices[:, 0]], uR[indices[:, 1]]], 1)

        logits = self.g(Z_pairs)
        if self.training:
            p_edges = LogitRelaxedBernoulli(temperature=self.temperature, logits=logits)
            A_vals = torch.sigmoid(p_edges.rsample())
        else:
            p_edges = Bernoulli(logits=logits)
            A_vals = p_edges.sample()

        # embed the values to the adjacency matrix
        A = torch.zeros(uM.size(0), uR.size(0), device=uM.device)
        A[indices[:, 0], indices[:, 1]] = A_vals.squeeze()

        return A

    @staticmethod
    def logitexp(logp):
        # https: // github.com / pytorch / pytorch / issues / 4007
        pos = torch.clamp(logp, min=-0.69314718056)
        neg = torch.clamp(logp, max=-0.69314718056)
        neg_val = neg - torch.log(1 - torch.exp(neg))
        pos_val = -torch.log(torch.clamp(torch.expm1(-pos), min=1e-20))
        return pos_val + neg_val

    @staticmethod
    def order_z(z):
        # scalar ordering function
        if z.size(1) == 1:
            return z
        log_cdf = torch.sum(
            torch.log(0.5 + 0.5 * torch.erf(z / math.sqrt(2))), dim=1, keepdim=True
        )
        return log_cdf


class RepEncoder(nn.Module):
    def __init__(self, f, use_u_diff=False, use_x=False):
        super().__init__()
        self.f = f
        self.cat = ConcatLayer()
        self.use_u_diff = use_u_diff
        self.use_x = use_x

    def forward(self, u, uR, XR, yR_encoded):
        input_list = [yR_encoded]
        if self.use_x:
            input_list.append(XR.unsqueeze(0))

        ## If we look at differences in U values, we need to increase the dimension
        ## (each representative member looks different to each dependent member)
        if self.use_u_diff:
            u_diff = u.unsqueeze(1) - uR.unsqueeze(0)
            for i, x in enumerate(input_list):
                input_list[i] = x.unsqueeze(1)
            input_list.append(u_diff.unsqueeze(0))

        rep_R = self.f(self.cat(*input_list))
        return rep_R


# **************************
# Representation Poolers
# **************************
class AveragePooler(nn.Module):
    def __init__(
        self,
        dim_z,
    ):
        super().__init__()
        self.dim_z = dim_z

        # normalizes the graph such that inner products correspond to averages of the parents
        self.norm_graph = lambda x: x / (torch.sum(x, 1, keepdim=True) + 1e-8)

    def forward(self, rep_R, GA):
        W = self.norm_graph(GA)
        pz_all = torch.matmul(W, rep_R)
        return pz_all


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

    def forward(self, rep_R, GA):
        if not self.set_transformer:
            rep_pooled = GA.unsqueeze(0).unsqueeze(-1).mul(rep_R).sum(2)
            pz_all = self.pool_net(rep_pooled)
        else:
            pz_all = self.settrans(GA.unsqueeze(0).unsqueeze(-1).mul(rep_R))
        return pz_all


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


# ******************
# Utility Modules
# *****************
# Note : These modules should probably live elsewhere in the repo,
# since they have use outside of the FNP.


class MLP(nn.Sequential):
    """
    A Multi-layer perceptron of dense layers with non-linear activation layers
    """

    def __init__(self, in_features, hs, out_features, act=nn.ReLU, final=None):
        self.in_features = in_features
        self.out_features = out_features
        layers = []
        layers.append(nn.Linear(in_features, hs[0]))
        layers.append(act())
        for i in range(len(hs) - 1):
            layers.append(nn.Linear(hs[i], hs[i + 1]))
            layers.append(act())
        layers.append(nn.Linear(hs[-1], out_features))
        if final is not None:
            layers.append(final())
        super().__init__(*layers)


class SequentialVarg(nn.Sequential):
    """
    This subclass of torch.nn.Sequential allows for stacking modules which take
    and/or return multiple arguments.
    """

    def forward(self, *input):
        for module in self:
            if isinstance(input, tuple):
                input = module(*input)
            else:
                input = module(input)
        return input


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


class ConcatLayer(nn.Module):
    def __init__(self, input_idxs=None):
        super().__init__()
        self.input_idxs = input_idxs

    def forward(self, *args):
        ## Filter only to arguments we want to concatenate
        if self.input_idxs is not None:
            args = [args[i] for i in self.input_idxs]
        else:
            args = list(args)

        ## Get the maximum size of each tensor dimension
        ## and repeat any tensors which have a 1 in
        ## a dimension
        sizes = []
        for d, _ in enumerate(args[0].size()):
            sizes.append(max([arg.size(d) for arg in args]))
        for d in range(len(sizes) - 1):
            size = sizes[d]
            if size > 1:
                for i, _ in enumerate(args):
                    if args[i].size(d) == 1:
                        r = [1] * len(sizes)
                        r[d] = size
                        args[i] = args[i].repeat(*r)
                    elif args[i].size(d) < size:
                        raise ValueError(
                            "The sizes in ConcatLayer need to be either the same or 1."
                        )
        X = torch.cat(args, -1)
        return X


class Flatten(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        assert len(x.shape) > 1

        return x.view(x.shape[0], -1)


class UnFlatten(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        # assert len(x.shape) == 2
        return x.view(*x.shape[0:-1], *self.shape)


# *************************
# Probabilistic Encoders
# ************************
class NormalEncoder(nn.Module):
    """
    This module takes two tensors of equal shape, mean and logscale, which parameterize
    a Normal distribution
    """

    def __init__(self, minscale=None):
        super().__init__()
        self.minscale = minscale

    def forward(self, mean_z, logscale_z):
        if self.minscale is not None:
            logscale_z = torch.log(
                self.minscale + (1 - self.minscale) * F.softplus(logscale_z)
            )
        pz = Normal(mean_z, logscale_z.exp())
        return pz


class ConstantDist(nn.Module):
    """
    A distribution which is constant; meant only as a placeholder for sampling
    """

    def __init__(self, X):
        super().__init__()
        self.X = X

    def log_prob(self, X):
        if X is self.X:
            return 0
        else:
            return -np.inf

    def sample(self):
        return self.X

    def rsample(self):
        return self.sample()


class IdentityEncoder(nn.Module):
    """
    Takes a tensor and returns a constant distribution which has all mass on that tensor.
    """

    def __init__(self):
        super().__init__()
        self.X = None

    def forward(self, X):
        return ConstantDist(X)


########################
# Model-specific code
# **********************


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
