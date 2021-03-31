from bliss.datasets.sdss import StarStamper
from itertools import product
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import Bernoulli
from torch.distributions.relaxed_bernoulli import LogitRelaxedBernoulli
from bliss.utils import MLP, ConcatLayer
from sklearn.cluster import KMeans
from pytorch_lightning import LightningModule
from ..utils import SequentialVarg, SplitLayer, NormalEncoder
from torch.distributions import Normal
from torch.utils.data import DataLoader
from torch.optim import Adam


class FNP(nn.Module):
    """
    This is an implementation of the Functional Neural Process (FNP)
    from http://arxiv.org/abs/1906.08324.

    This implementation is based on https://github.com/AMLab-Amsterdam/FNP,
    distributed with the MIT license. Significant changes have been made,
    so any problems or bugs should not be attributed to the original authors.

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

    def encode(self, XR, yR, XM, G_in=None, A_in=None):
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
        if G_in is None:
            G = self.dep_graph.sample_G(uR)
        else:
            G = G_in

        GA = torch.cat([G, A], 0)
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
        # pylint: disable=attribute-defined-outside-init
        """
        Calculates the log difference between pz and qz (with an optional free bits strategy)
        """
        pqz_all = pz.log_prob(z) - qz.log_prob(z)
        assert torch.isnan(pqz_all).sum() == 0
        if self.fb_z > 0:
            log_qpz = -torch.sum(pqz_all)

            if self.training:
                if log_qpz.item() > self.fb_z * z.size(0) * z.size(1) * (1 + 0.05):
                    self.lambda_z = torch.clamp(self.lambda_z * (1 + 0.1), min=1e-8, max=1.0)
                elif log_qpz.item() < self.fb_z * z.size(0) * z.size(1):
                    self.lambda_z = torch.clamp(self.lambda_z * (1 - 0.1), min=1e-8, max=1.0)

            log_pqz = self.lambda_z * pqz_all.sum()

        else:
            log_pqz = pqz_all.sum()
        return log_pqz

    def forward(self, XR, yR, XM, yM, G_in=None, A_in=None):
        return -self.log_prob(XR, yR, XM, yM, G_in, A_in)

    def predict(self, x_new, XR, yR, sample=True, A_in=None, sample_Z=True):
        n_ref = XR.size(0)
        u, pz = self.encode(XR, yR, x_new, None, A_in)
        uM = u[n_ref:]
        if sample_Z:
            z = pz.rsample()
        else:
            z = pz.mean
        zM = z[:, n_ref:]

        py = self.label_vdecoder(zM, uM.unsqueeze(0))
        if sample:
            y_pred = py.sample()
        else:
            y_pred = py.mean

        return y_pred


class HNP(nn.Module):
    """
    This is an implementation of the Hierarchical Neural Process (HNP), a new model.
    """

    def __init__(
        self,
        dep_graph,
        z_inference,
        z_pooler,
        h_prior,
        h_pooler,
        y_decoder,
        fb_z=0.0,
    ):
        super().__init__()
        ## Learned Submodules
        self.dep_graph = dep_graph
        self.z_inference = z_inference
        self.z_pooler = z_pooler
        self.h_prior = h_prior
        self.h_pooler = h_pooler
        self.y_decoder = y_decoder
        ## Initialize free-bits regularization
        self.fb_z = fb_z
        self.register_buffer("lambda_z", torch.tensor(1e-8))

    def encode(self, X, S):
        n_inputs = S.size(0)

        ## Calculate dependency graph
        G = self.dep_graph(X)

        ## Calculate the prior distribution for the H
        pH = self.h_prior(X, G)

        if n_inputs > 0:
            ## Encode the available stamps
            Zi = self.z_inference(X[:n_inputs], S)
            ## Sample the hierarchical latent variables from the latent variables
            # qH = self.h_pooler(X, G, Zi)
            qH = self.h_pooler(Zi, G[:n_inputs].transpose(1, 0))
        else:
            qH = pH
        H = qH.rsample()
        ## Conditional on the H, calculate  Z
        Z = self.z_pooler(H, G)

        ## Calculate predicted stamp
        pY = self.y_decoder(Z, X)

        return pH, qH, H, Z, pY

    def log_prob(self, X, S, Y):
        n_inputs = S.size(0)
        pH, qH, H, _, pY = self(X, S)
        log_pqh = pH.log_prob(H) - qH.log_prob(H)
        log_y = pY.log_prob(Y)
        elbo = log_pqh.sum() + log_y.sum()
        return elbo

    def forward(self, X, S):
        return self.encode(X, S)

    def predict(self, X, S, mean_Y=False, cond_output=False):
        n_inputs = S.size(0)
        pY = self.encode(X, S)[4]
        if mean_Y:
            Y = pY.loc.detach()
        else:
            Y = pY.sample()
        if cond_output:
            Y[:n_inputs] = S
        return Y


class KMeansDepGraph(nn.Module):
    def __init__(self, n_clusters):
        super().__init__()
        self.n_clusters = n_clusters

    def forward(self, X):
        km = KMeans(n_clusters=self.n_clusters)
        c = km.fit_predict(X.cpu().numpy())
        G = torch.zeros((len(c), self.n_clusters)).to(X.device)
        for i in range(self.n_clusters):
            G[c == i, i] = 1.0
        return G


## ***********************
## FNP-specific submodules
## ***********************
class DepGraph(nn.Module):
    """
    A dependency-graph module for use within FNP.
    For tensors of input encodings from the reference points (uR)
    and dependent points (uM), this module returns matrices
    that represent the dependency structure.

    This implementation is based on https://github.com/AMLab-Amsterdam/FNP,
    distributed with the MIT license. Significant changes have been made,
    so any problems or bugs should not be attributed to the original authors.
    """

    def __init__(self, dim_u, temperature=0.3):
        super().__init__()
        ## Dimension of the encoded-input space
        self.dim_u = dim_u
        ## Temperature for LogitRelaxedBernoulli when training
        self.temperature = temperature
        ## Learned parameter for self.g, the pairwise distance function
        self.g_logscale = nn.Parameter(torch.tensor(np.log(self.dim_u) * 0.5))

    def sample_G(self, uR):
        # get the indices of an upper triangular adjacency matrix that represents the DAG
        idx_utr = np.triu_indices(uR.size(0), 1)

        # get the ordering
        ordering = self.order_z(uR)
        # sort the latents according to the ordering
        sort_idx = torch.sort(torch.squeeze(ordering), 0)[1]
        Y = uR[sort_idx, :]
        # form the latent pairs for the edges, and
        # get the logits for the edges in the DAG
        logits = self.g(Y[idx_utr[0]], Y[idx_utr[1]])

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
        logits = self.g(uM[indices[:, 0]], uR[indices[:, 1]])
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

    def g(self, z1, z2):
        sq_norm2 = (z2 - z1).pow(2)
        a = -0.5 * sq_norm2.sum(1, keepdim=True) / self.g_logscale.exp()
        b = self.logitexp(a).view(z1.size(0), 1)
        return b

    @staticmethod
    def logitexp(logp):
        pos = torch.clamp(logp, min=-0.69314718056)
        neg = torch.clamp(logp, max=-0.69314718056)
        neg_val = neg - torch.log1p(-torch.exp(neg))
        pos_val = -torch.log(torch.clamp(torch.expm1(-pos), min=1e-20))
        return pos_val + neg_val

    @staticmethod
    def order_z(z):
        # scalar ordering function
        if z.size(1) == 1:
            return z
        log_cdf = torch.sum(torch.log(0.5 + 0.5 * torch.erf(z / np.sqrt(2))), dim=1, keepdim=True)
        return log_cdf


class RepEncoder(nn.Module):
    """
    Module which forms an encoded representation based on
    the encoded input (U), the reference input (xR),
    and the encoded reference labels (yR).
    """

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
    """
    Pools together representations by taking a sum.
    """

    def __init__(
        self,
        dim_z,
        f=None,
    ):
        super().__init__()
        self.dim_z = dim_z
        self.f = f

        # normalizes the graph such that inner products correspond to averages of the parents
        self.norm_graph = lambda x: x / (torch.sum(x, 1, keepdim=True) + 1e-8)

    def forward(self, rep_R, GA):
        W = self.norm_graph(GA)
        if self.f:
            rep_R = self.f(rep_R)
        pz_all = torch.matmul(W, rep_R)
        return pz_all


class SetPooler(nn.Module):
    """
    Pools together representations using a set transformer architecture.
    See https://arxiv.org/abs/1810.00825.
    """

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
                MLP(dim_in, self.pooling_layers, 2 * self.dim_z),
            )

    def forward(self, rep_R, GA):
        if not self.set_transformer:
            rep_pooled = GA.unsqueeze(0).unsqueeze(-1).mul(rep_R).sum(2)
            pz_all = self.pool_net(rep_pooled)
        else:
            pz_all = self.settrans(GA.unsqueeze(0).unsqueeze(-1).mul(rep_R))
        return pz_all


# *********************
# Set Transformer
# *********************
# This implementation is taken from https://github.com/juho-lee/set_transformer/,
# which comes with the MIT license.


class MAB(nn.Module):
    """
    A Multihead Attention Block for use in a set transformer.
    See https://arxiv.org/abs/1810.00825.

    This implementation is from https://github.com/juho-lee/set_transformer/,
    which comes with the MIT license.
    """

    def __init__(self, dim_Q, dim_K, dim_V, num_heads, ln=False):
        super().__init__()
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

        A = torch.softmax(Q_.matmul(K_.transpose(-2, -1)) / np.sqrt(self.dim_V), -1)
        O = torch.cat((Q_ + A.matmul(V_)).split(Q.size(-3), -3), -1)
        O = O if getattr(self, "ln0", None) is None else self.ln0(O)
        O = O + F.relu(self.fc_o(O))
        O = O if getattr(self, "ln1", None) is None else self.ln1(O)
        return O.squeeze(-3)


class SAB(nn.Module):
    """
    A Self-Attention Block for use in a set transformer.
    See https://arxiv.org/abs/1810.00825.

    This implementation is from https://github.com/juho-lee/set_transformer/,
    which comes with the MIT license.
    """

    def __init__(self, dim_in, dim_out, num_heads, ln=False):
        super().__init__()
        self.mab = MAB(dim_in, dim_in, dim_out, num_heads, ln=ln)

    def forward(self, X):
        return self.mab(X, X)


class PMA(nn.Module):
    """
    A Pooling by Multihead Attention block for use in a set transformer.
    See https://arxiv.org/abs/1810.00825.

    This implementation is from https://github.com/juho-lee/set_transformer/,
    which comes with the MIT license.
    """

    def __init__(self, dim, num_heads, num_seeds, ln=False, squeeze_out=False):
        super().__init__()
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


class StarHNP(HNP):
    def __init__(self, stampsize, dz=4, fb_z=0.0, n_clusters=5):
        dy = stampsize ** 2
        dep_graph = KMeansDepGraph(n_clusters=n_clusters)
        z_inference = SequentialVarg(ConcatLayer([1]), MLP(dy, [16, 8], dz))

        z_pooler = AveragePooler(dz)

        h_prior = lambda X, G: Normal(
            torch.zeros(G.size(1), dz, device=G.device), torch.ones(G.size(1), dz, device=G.device)
        )

        h_pooler = SimpleHPooler(dz)

        y_decoder = SequentialVarg(
            ConcatLayer([0]),
            MLP(dz, [8, 16, 32], 2 * dy),
            SplitLayer(dy, -1),
            NormalEncoder(minscale=1e-7),
        )
        super().__init__(dep_graph, z_inference, z_pooler, h_prior, h_pooler, y_decoder, fb_z)


class SDSS_HNP(LightningModule):
    def __init__(self, stampsize=5, dz=4, sdss_dataset=None, max_cond_inputs=1000, n_clusters=5):
        super().__init__()
        self.sdss_dataset = sdss_dataset
        self.stamper = StarStamper(stampsize)
        self.max_cond_inputs = max_cond_inputs
        self.n_clusters = n_clusters
        self.hnp = StarHNP(stampsize, dz, fb_z=0.0, n_clusters=n_clusters)
        self.valid_losses = []

    def training_step(self, batch, batch_idx):
        X, S, YY = self.prepare_batch(batch)
        loss = -self.hnp.log_prob(X, S, YY) / X.size(0)
        return loss

    def prepare_batch(self, batch, num_cond_inputs=None):
        X, img, locs = batch
        if not isinstance(img, torch.Tensor):
            img = torch.from_numpy(img)
        if not isinstance(locs, torch.Tensor):
            locs = torch.from_numpy(locs)
        YY = self.stamper(img, locs[:, 1], locs[:, 0])[0]
        YY = YY.reshape(-1, 25)
        YY = (YY - YY.mean(1, keepdim=True)) / YY.std(1, keepdim=True)
        if num_cond_inputs is None:
            num_cond_inputs = self.max_cond_inputs
        S = YY[: min(YY.size(0), num_cond_inputs)]
        return X, S, YY

    def predict(self, X, S):
        out = self.hnp.predict(X, S, mean_Y=True)
        return out

    def validation_step(self, batch, batch_idx):
        loss = self.training_step(batch, batch_idx)
        self.log("val_loss", loss)
        self.valid_losses.append(loss.item())
        return loss

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=1e-3)

    def train_dataloader(self):
        return DataLoader(self.sdss_dataset, batch_size=None, batch_sampler=None)


class SimpleHPooler(nn.Module):
    def __init__(self, dh):
        super().__init__()
        self.ap = AveragePooler(dh)

    def forward(self, Z, G):
        z_pooled = self.ap(Z, G)
        precis = 1.0 + G.sum(1)
        std = precis.reciprocal().sqrt().unsqueeze(1).repeat(1, z_pooled.size(1))
        return Normal(z_pooled, std)
