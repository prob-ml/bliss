import os
import subprocess as sp
from shutil import which
import time
from pytorch_lightning import LightningModule
import torch
import math
import argparse
import numpy as np
import torch.nn as nn
from pathlib import Path
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from torch.distributions import MultivariateNormal
from torch.optim import Adam
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset

from bliss.models.fnp import (
    DepGraph,
    FNP,
    AveragePooler,
    SetPooler,
    RepEncoder,
    HNP,
)

from bliss.utils import MLP, SequentialVarg, SplitLayer, ConcatLayer, NormalEncoder
from utils import IdentityEncoder, Conv2DAutoEncoder, ReshapeWrapper
from rotate import PsfFnpData
from torch.distributions import Normal


class RotatingStarHNP(LightningModule):
    def __init__(
        self,
        dim_z=8,
        dim_h=8,
        fb_z=0.0,
        output_layers=[128],
        pooling_layers=[64],
        st_numheads=[2, 2],
        size_h=10,
        size_w=10,
        kernel_sizes=[3, 3],
        strides=[1, 1],
        conv_channels=[20, 20],
    ):
        self.dim_h = dim_h
        self.dim_z = dim_z
        dep_graph = self._make_dep_graph
        conv_autoencoder = Conv2DAutoEncoder(
            size_h,
            size_w,
            conv_channels,
            kernel_sizes,
            strides,
            last_decoder_channel=1,
        )
        z_inference = nn.Sequential(
            conv_autoencoder.encoder, nn.Linear(conv_autoencoder.dim_rep, self.dim_z)
        )
        z_pooler = RotatingZPooler(self.dim_h, self.dim_z, pooling_layers, True, st_numheads)

        h_prior = lambda X, G: Normal(
            torch.zeros(X.size(0), self.dim_h, device=X.device),
            torch.ones(X.size(0), self.dim_h, device=X.device),
        )

        h_pooler = RotatingHPooler(self.dim_z, self.dim_h, pooling_layers, st_numheads)

        y_decoder = nn.Sequential(
            MLP(self.dim_z, output_layers, conv_autoencoder.dim_rep), conv_autoencoder.decoder
        )

        self.hnp = HNP(
            dep_graph,
            z_inference,
            z_pooler,
            h_prior,
            h_pooler,
            y_decoder,
            fb_z=fb_z,
        )

        self.valid_losses = []

    @staticmethod
    def _make_dep_graph(X):
        G = torch.zeros(X.size(0), X.size(0), device=X.device)
        G[0, 0] = 0.5
        G[0, 1] = 0.5
        G[-1, -2] = 0.5
        G[-1, -1] = 0.5
        for i in range(1, G.size(0) - 1):
            G[i, i - 1] = 1 / 3
            G[i, i] = 1 / 3
            G[i, i + 1] = 1 / 3
        return G

    def training_step(self, batch, batch_idx, valid=False):
        X, S, Y = self.prepare_batch(batch)
        elbo = self.hnp.log_prob(X, S, Y) / X.size(0)
        loss = -elbo
        if not valid:
            self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.training_step(batch, batch_idx, valid=True)
        self.log("val_loss", loss)
        self.valid_losses.append(loss.item())

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=1e-3)

    # def predict(self, x_new, XR, yR, sample=True, A_in=None, sample_Z=True):
    #     y_pred = super().predict(x_new, XR, yR, sample=sample, A_in=A_in, sample_Z=sample_Z)
    #     if self.transf_y is not None:
    #         y_pred = self.inverse_transform(y_pred)
    #     return y_pred

    # def inverse_transform(self, y):
    #     y = y.squeeze(-3)
    #     y_flat = y.reshape(*y.shape[0:2], -1).cpu().detach().numpy()
    #     y_flat_invt = self.transf_y.inverse_transform(y_flat)
    #     y_out = torch.from_numpy(y_flat_invt).resize_(y.size())
    #     return y_out


class RotatingZPooler(nn.Module):
    def __init__(self, dim_x, dim_h, dim_z, pooling_layers, st_numheads):
        super().__init__()
        self.dim_x = dim_x
        self.dim_h = dim_h
        self.dim_z = dim_z
        self.pooler = SetPooler(self.dim_x + self.dim_h, dim_z, pooling_layers, True, st_numheads)
        self.cat = ConcatLayer()

    def forward(self, X, X_H, H, G):
        X_diff = X.unsqueeze(1) - X_H.unsqueeze(0)
        R = self.cat(X_diff, H.unsqueeze(0))
        Z = self.pooler(R, G)
        return Z


class RotatingHPooler(nn.Module):
    def __init__(self, dim_z, dim_h, pooling_layers, st_numheads):
        super().__init__()
        self.dim_z = dim_z
        self.dim_h = dim_h
        self.pooler = SetPooler(self.dim_z, 2 * self.dim_h, pooling_layers, True, st_numheads)
        self.cat = ConcatLayer()
        self.normal_encoder = NormalEncoder(minscale=1e-7)

    def forward(self, X, X_H, Z, GT):
        X_diff_T = X.unsqueeze(0) - X_H.unsqueeze(1)
        R = self.cat(X_diff_T, Z.unsqueeze(0))
        H_rep = self.pooler(R, GT)
        mu_H, logstd_H = torch.split(H_rep, self.dim_h, -1)
        qH = self.normal_encoder(mu_H, logstd_H)
        return qH


outdir = Path(__file__).parent.joinpath("output_hnp")


def main(args):
    print("Setting CUDA device to", args.cuda_device[0])
    torch.cuda.set_device(args.cuda_device[0])

    epochs_outer = args.epochs[0]
    generate = not args.nogenerate
    n_ref = args.n_ref[0]
    N = args.N[0]
    size_h = args.size_h[0]
    size_w = args.size_w[0]
    cov_multiplier = args.cov_multiplier[0]
    base_angle = math.pi * args.base_angle[0]
    angle_stdev = math.pi * args.angle_stdev[0]

    K_probs = torch.tensor([0.25] * 4)
    Ks = torch.multinomial(K_probs, n_ref - 1, replacement=True) + 1

    batch_size = args.batch_size[0]
    learnrate = args.learnrate[0]

    if not outdir.exists():
        outdir.mkdir(parents=True)

    if generate:
        print("Generating data...")
        tic = time.perf_counter()
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
            device="cuda:{0}".format(args.cuda_device[0]),
        )
        star_data.export_images(outdir.joinpath("rotating_star_dgp.png"), nrows=10)
        star_data.export_images(
            outdir.joinpath("rotating_star_dgp_valid.png"), nrows=10, valid=True
        )
        torch.save(star_data, outdir.joinpath("star_data.pt"))
        toc = time.perf_counter()
        print("DONE (time = {:0.4f} seconds)".format(toc - tic))
    else:
        star_data = torch.load(outdir.joinpath("star_data.pt"))
        n_ref = args.n_ref[0] = star_data.n_ref
        N = args.N[0] = star_data.images.size(0)
        size_h = args.size_h[0] = star_data.images.size(2)
        size_w = args.size_w[0] = star_data.images.size(3)
        cov_multiplier = args.cov_multiplier[0] = star_data.dgp.cov_multiplier
        base_angle = args.base_angle[0] = star_data.dgp.base_angle
        angle_stdev = args.angle_stdev[0] = star_data.dgp.angle_stdev

        args.base_angle[0] /= math.pi
        args.angle_stdev[0] /= math.pi

    print("Arguments:")
    print(args)
    torch.save(vars(args), outdir.joinpath("args.pt"))

    if not args.skiptrain:
        train_dataloader = DataLoader()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run Point Spread Function (PSF) using Functional Neural Process (FNP)."
    )

    parser.add_argument(
        "--case",
        type=str,
        default="rotate",
        choices=["rotate"],
        help="Which case study to run",
    )

    parser.add_argument(
        "--device",
        dest="cuda_device",
        type=int,
        nargs=1,
        help="CUDA device to run on",
        default=[0],
    )
    parser.add_argument(
        "--epochs",
        dest="epochs",
        type=int,
        nargs=1,
        help="Number of epochs to run through the data",
        default=[100],
    )
    parser.add_argument(
        "--batchsize",
        dest="batch_size",
        type=int,
        nargs=1,
        help="Minibatch size for SGD",
        default=[10],
    )
    parser.add_argument(
        "--learnrate",
        dest="learnrate",
        type=float,
        nargs=1,
        help="Learning rate for ADAM",
        default=[1e-3],
    )
    parser.add_argument("--skiptrain", dest="skiptrain", action="store_true")

    parser.add_argument(
        "--nogenerate", help="Generate the data rather than read it in", action="store_true"
    )
    parser.add_argument(
        "--n_ref",
        help="(DGP) Number of reference stars to add",
        type=int,
        dest="n_ref",
        nargs=1,
        default=[50],
    )
    parser.add_argument(
        "--N_rows",
        help="(DGP) Number of independent rows to generate",
        type=int,
        dest="N",
        nargs=1,
        default=[1000],
    )
    parser.add_argument(
        "--sizeh",
        help="(DGP) Height of generated images",
        type=int,
        dest="size_h",
        nargs=1,
        default=[50],
    )
    parser.add_argument(
        "--sizew",
        help="(DGP) Width of generated images",
        type=int,
        dest="size_w",
        nargs=1,
        default=[50],
    )
    parser.add_argument(
        "--covmultiplier",
        help="(DGP) How big the star is",
        type=float,
        dest="cov_multiplier",
        nargs=1,
        default=[20.0],
    )
    parser.add_argument(
        "--baseangle",
        help="(DGP) How fast the stars rotate on average (as a multiple of pi)",
        type=float,
        dest="base_angle",
        nargs=1,
        default=[(3 / 23)],
    )
    parser.add_argument(
        "--anglestdev",
        help="(DGP) The standard deviation of rotation (as a multiple of pi; higher means angle is less predictable)",
        type=float,
        dest="angle_stdev",
        nargs=1,
        default=[0.025],
    )

    args = parser.parse_args()
    main(args)
