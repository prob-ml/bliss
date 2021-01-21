import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
import math
import os
import sys
import time

# path = os.path.abspath("..")
# if path not in sys.path:
#     sys.path.insert(0, path)

from sklearn.preprocessing import StandardScaler
from torch.optim import Adam
from importlib import reload

import case_studies.fnp.rotate_dgp as dgp
from case_studies.fnp.rotate_dgp import PsfFnpData

import bliss.models.fnp as fnp

# outdir = os.environ["BLISS_DIR"]+"case_studies/fnp/"
outdir = "/home/derek/projects/bliss/case_studies/fnp/"


parser = argparse.ArgumentParser(
    description="Run Point Spread Function (PSF) using Functional Neural Process (FNP)."
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
    default=[1000],
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
parser.add_argument("--condref", dest="condition_on_ref", action="store_true")
parser.add_argument(
    "--notrainproposal", dest="train_separate_proposal", action="store_true"
)
parser.add_argument("--skiptrain", dest="skiptrain", action="store_true")

parser.add_argument(
    "--generate", help="Generate the data rather than read it in", action="store_true"
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
    default=[500],
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
    default=[0.1],
)

if __name__ == "__main__":
    if torch.get_num_threads() > 8:
        torch.set_num_threads(8)
    args = parser.parse_args()
    print("Setting CUDA device to", args.cuda_device[0])
    torch.cuda.set_device(args.cuda_device[0])

    epochs_outer = args.epochs[0]
    condition_on_ref = args.condition_on_ref
    train_separate_proposal = not args.train_separate_proposal
    generate = args.generate
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

    if generate:
        print("Generating data...")
        tic = time.perf_counter()
        star_data = dgp.PsfFnpData(
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
        os.remove(outdir + "rotating_star_dgp.png")
        os.remove(outdir + "rotating_star_dgp_valid.png")
        os.remove("../temp/star_data.pt")
        star_data.export_images(outdir + "rotating_star_dgp.png", nrows=10)
        star_data.export_images(
            outdir + "rotating_star_dgp_valid.png", nrows=10, valid=True
        )
        torch.save(star_data, outdir + "star_data.pt")
        toc = time.perf_counter()
        print("DONE (time = {:0.4f} seconds)".format(toc - tic))
    else:
        star_data = torch.load(outdir + "star_data.pt")
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
    torch.save(vars(args), outdir + "args.pt")

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
        num_M=star_data.X_m.size(0),
        dim_z=8,
        fb_z=1.0,
        use_plus=False,
        G=star_data.G,
        A=star_data.A,
        mu_nu_layers=[128, 64, 32],
        use_x_mu_nu=False,
        use_direction_mu_nu=True,
        output_layers=[32, 64, 128],
        x_as_u=True,
        condition_on_ref=condition_on_ref,
        train_separate_proposal=train_separate_proposal,
        train_separate_extrapolate=False,
        discrete_orientation=False,
        set_transformer=True,
    )

    if torch.cuda.is_available():
        print("Moving data to GPU {}".format(args.cuda_device[0]))
        tic = time.perf_counter()
        star_data.cuda()
        fnp_model = fnp_model.cuda()
        toc = time.perf_counter()
        print("DONE (time = {:0.4f} seconds)".format(toc - tic))

    optimizer = Adam(fnp_model.parameters(), lr=learnrate)
    fnp_model.train()

    skip_train = args.skiptrain

    if not skip_train:
        print("Running with {0} outer epochs".format(epochs_outer))
        callback = 100
        epochs = epochs_outer * star_data.N
        nbatches = star_data.N // batch_size
        tic = time.perf_counter()
        for i in range(epochs_outer):
            points = torch.randperm(star_data.N)
            for i_n in range(nbatches):
                ns = points[(batch_size * i_n) : (batch_size * (i_n + 1))]
                optimizer.zero_grad()
                y_r_in = star_data.y_r[ns]
                y_m_in = star_data.y_m[ns]
                loss = (
                    fnp_model(star_data.X_r, y_r_in, star_data.X_m, y_m_in) / batch_size
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
        toc = time.perf_counter()
        print("Done training (time = {:0.4f} seconds)".format(toc - tic))

        # os.remove("../temp/epochs_outer.pt")
        # os.remove("../temp/fnp_model.pt")
        torch.save(epochs_outer, outdir + "epochs_outer.pt")
        torch.save(fnp_model.state_dict(), outdir + "fnp_model.pt")
    else:
        fnp_model.load_state_dict(torch.load(outdir + "fnp_model.pt"))

    X_extra = star_data.X_ref.max() + torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0]).cpu()
    X_extra_std = torch.from_numpy(
        star_data.stdx.transform(X_extra.cpu().unsqueeze(1)).astype(np.float32)
    ).cuda()
    A_extra = torch.zeros(5, fnp_model.A.size(1)).cuda()
    A_extra[:, -1] = 1
    X_pred = torch.cat([star_data.X_dep, X_extra])
    X_pred_std = torch.cat([star_data.X_m, X_extra_std])
    A_pred = torch.cat([fnp_model.A.cuda(), A_extra])
    myimg = star_data.make_fnp_mean_image(
        fnp_model, X=X_pred_std, X_nostd=X_pred, A=A_pred, samples=10, valid=True
    )
    vmin = star_data.images.min()
    vmax = star_data.images.max()
    os.remove(outdir + "rotating_star_fnp_dgp_valid.png")
    plt.imsave(outdir + "rotating_star_fnp_dgp_valid.png", myimg, vmin=vmin, vmax=vmax)

    myimgvar = star_data.make_fnp_var_image(fnp_model, samples=10, valid=True)

    os.remove(outdir + "rotating_star_fnp_dgp_valid_var.png")
    plt.imsave(
        outdir + "rotating_star_fnp_dgp_valid_var.png", myimgvar, vmin=vmin, vmax=vmax
    )

    many_images = torch.stack(
        [star_data.make_fnp_single_image(fnp_model, valid=True) for i in range(100)]
    )
    for i in range(many_images.size(0)):
        imgname = outdir + "fnp_valid_{:02d}.png".format(i)
        if os.path.isfile(imgname):
            os.remove(imgname)
        plt.imsave(imgname, many_images[i], vmin=vmin, vmax=vmax)

    mean_Z_img = star_data.make_fnp_single_image(fnp_model, valid=True, sample_Z=False)
    meanz_file = outdir + "fnp_valid_mean.png"
    if os.path.isfile(meanz_file):
        os.remove(meanz_file)
    plt.imsave(meanz_file, mean_Z_img, vmin=vmin, vmax=vmax)

    myimg = star_data.make_fnp_mean_image(fnp_model, samples=10, valid=False, N=10)
    vmin = star_data.images.min()
    vmax = star_data.images.max()
    os.remove(outdir + "rotating_star_fnp_dgp_train.png")
    plt.imsave(outdir + "rotating_star_fnp_dgp_train.png", myimg, vmin=vmin, vmax=vmax)
