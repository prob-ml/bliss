#!/usr/bin/env python
import argparse
import json
import os
from pathlib import Path
import numpy as np
import torch

from celeste import train, psf_transform
from celeste.models import sourcenet
from celeste.datasets import simulated_datasets


def load_psf(paths, device):
    psf_file = paths["data"].joinpath("fitted_powerlaw_psf_params.npy")
    psf_params = torch.tensor(np.load(psf_file), device=device)
    power_law_psf = psf_transform.PowerLawPSF(psf_params)
    psf = power_law_psf.forward().detach()

    return psf


def load_background(data_params, device):
    background = torch.zeros(
        data_params["n_bands"], data_params["slen"], data_params["slen"], device=device
    )
    background[0] = 686.0
    background[1] = 1123.0
    return background


def setup_paths(args):
    root_path = Path(args.root_dir)
    path_dict = {
        "root": root_path,
        "data": root_path.joinpath("data"),
        "config": root_path.joinpath("config"),
        "results": root_path.joinpath("results"),
    }

    for p in path_dict.values():
        assert p.exists(), f"path {p.as_posix()} does not exist"

    return path_dict


def setup_device(args):
    assert (
        args.no_cuda or torch.cuda.is_available()
    ), "cuda is not available but --no-cuda is false"

    if not args.no_cuda:
        device = torch.device(f"cuda:{args.device}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    return device


def load_data_params_from_args(params_file, args):
    with open(params_file, "r") as fp:
        data_params = json.load(fp)

    args_dict = vars(args)
    for k in data_params:
        if k in args_dict and args_dict[k] is not None:
            data_params[k] = args_dict[k]
    return data_params


def main(args):

    paths = setup_paths(args)
    device = setup_device(args)

    print(
        f"running sleep phase for n_epochs={args.n_epochs}, batchsize={args.batchsize}, "
        f"n_images={args.n_images}"
    )

    data_param_file = paths["data"].joinpath(
        "dataset_params/default_galaxy_parameters.json"
    )
    data_params = load_data_params_from_args(data_param_file, args)
    background_file = paths["data"].joinpath(data_params["background_file"])

    print("data params to be used:", data_params)
    print("background file:", background_file)

    galaxy_dataset = simulated_datasets.GalaxyDataset.load_dataset_from_params(
        args.n_images, data_params, background_file
    )

    galaxy_encoder = sourcenet.SourceEncoder(
        slen=data_params["slen"],
        n_bands=data_params["n_bands"],
        ptile_slen=args.ptile_slen,
        step=args.step,
        edge_padding=args.edge_padding,
        max_detections=args.max_detections,
        n_source_params=galaxy_dataset.simulator.latent_dim,
    ).to(device)

    out_dir = paths["results"].joinpath(args.output_name)
    train_sleep = train.SleepTraining(
        galaxy_encoder,
        galaxy_dataset,
        data_params["slen"],
        num_bands=1,
        n_source_params=galaxy_dataset.simulator.latent_dim,
        batchsize=args.batchsize,
        eval_every=args.print_every,
        out_dir=out_dir,
        seed=pargs.seed,
    )

    print("training starting...")
    train_sleep.run(args.n_epochs)


if __name__ == "__main__":

    # Setup arguments.
    parser = argparse.ArgumentParser(
        description="Sleep phase galaxy training [argument parser]",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--device", type=int, default=0, metavar="DEV", help="GPU device ID"
    )

    parser.add_argument(
        "--root-dir",
        help="Absolute path to directory containing bin and celeste package.",
        type=str,
        default=os.path.abspath(".."),
    )

    parser.add_argument(
        "--output-name",
        type=str,
        default="test",
        metavar="DIR",
        help="Directory name relative to root/results path, where output will be saved.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        metavar="S",
        help="Random seed for tensor flow cuda.",
    )
    parser.add_argument(
        "--no-cuda",
        action="store_true",
        help="whether to using a discrete graphics card or not.",
    )

    # data params that can be changed, default==None means use ones in .json file.
    parser.add_argument("--slen", type=int, default=None)
    parser.add_argument("--max-galaxies", type=int, default=None)
    parser.add_argument("--mean-galaxies", type=int, default=None)

    # training params
    parser.add_argument(
        "--n-images", type=int, default=640, help="Number of images in epoch"
    )
    parser.add_argument(
        "--n-epochs", type=int, default=201, help="Number of epochs to run for."
    )
    parser.add_argument(
        "--batchsize", type=int, default=32, help="Number of batches in each epoch."
    )
    parser.add_argument(
        "--print-every",
        type=int,
        default=20,
        help="Log every {print_every} number of times",
    )

    parser.add_argument(
        "--ptile-slen",
        type=int,
        default=20,
        help="Side length of the padded tile in pixels.",
    )
    parser.add_argument(
        "--step", type=int, default=5, help="Distance between tile centers in pixels."
    )
    parser.add_argument(
        "--edge-padding",
        type=int,
        default=5,
        help="Padding around each tile in pixels.",
    )
    parser.add_argument(
        "--max-detections",
        type=int,
        default=2,
        help="Number of max detections in each tile. ",
    )

    pargs = parser.parse_args()
    main(pargs)
