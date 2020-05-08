#!/usr/bin/env python

import json
import argparse
import numpy as np
import torch
import torch.optim as optim

from celeste import sleep_lib
from celeste.models import sourcenet_lib
from celeste.datasets import simulated_datasets_lib
from celeste.utils import const


def set_seed(seed):
    np.random.seed(65765)
    _ = torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_data_params(args):
    parameters_path = const.data_path.joinpath("default_galaxy_parameters.json")
    with open(parameters_path, "r") as fp:
        data_params = json.load(fp)

    args_dict = vars(args)
    for k in data_params:
        if k in args_dict and args_dict[k] is not None:
            data_params[k] = args_dict[k]
    return data_params


def get_optimizer(galaxy_encoder):
    learning_rate = 1e-3
    weight_decay = 1e-5
    optimizer = optim.Adam(
        [{"params": galaxy_encoder.parameters(), "lr": learning_rate}],
        weight_decay=weight_decay,
    )
    return optimizer


def prepare_filepaths(results_dir):
    out_dir = const.results_path.joinpath(results_dir)
    out_dir.mkdir(exist_ok=True, parents=True)

    state_dict_file = out_dir.joinpath("galaxy_i.dat")
    print(f"state dict file: {state_dict_file.as_posix()}")

    output_file = out_dir.joinpath(
        "output.txt"
    )  # save the output that is being printed.
    print(f"output file: {output_file.as_posix()}")

    return state_dict_file, output_file


def train(
    galaxy_encoder,
    dataset,
    optimizer,
    n_epochs,
    batchsize,
    print_every,
    state_dict_file,
    output_file,
):
    print("training...")

    sleep_phase = sleep_lib.GalaxySleep(
        galaxy_encoder,
        dataset,
        n_epochs,
        galaxy_encoder.n_source_params,
        state_dict_file=state_dict_file,
        output_file=output_file,
        optimizer=optimizer,
        batchsize=batchsize,
        print_every=print_every,
    )

    print(f"running sleep phase for n_epochs={n_epochs}, batchsize={batchsize}")
    sleep_phase.run_sleep()


def main(pargs):

    const.set_device(pargs.device)  # set global device to use.

    set_seed(pargs.seed)
    data_params = load_data_params(pargs)

    state_dict_file, output_file = prepare_filepaths(pargs.results_dir)

    # setup dataset.
    galaxy_dataset = simulated_datasets_lib.GalaxyDataset.load_dataset_from_params(
        pargs.n_images, data_params
    )

    galaxy_encoder = sourcenet_lib.SourceEncoder(
        slen=data_params["slen"],
        n_bands=data_params["n_bands"],
        ptile_slen=pargs.ptile_slen,
        step=pargs.step,
        edge_padding=pargs.edge_padding,
        max_detections=pargs.max_detections,
        n_source_params=galaxy_dataset.simulator.latent_dim,
    ).to(const.device)

    optimizer = get_optimizer(galaxy_encoder)

    train(
        galaxy_encoder,
        galaxy_dataset,
        optimizer,
        pargs.n_epochs,
        pargs.batchsize,
        pargs.print_every,
        state_dict_file,
        output_file,
    )


if __name__ == "__main__":

    # Setup arguments.
    parser = argparse.ArgumentParser(
        description="Sleep phase galaxy training [argument parser]"
    )

    parser.add_argument(
        "--device", type=int, default=0, metavar="DEV", help="GPU device ID"
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="test",
        metavar="DIR",
        help="Directory in results path, where output will be saved.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Whether to overwrite --results-dir if directory already exists or throw an error",
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
        help="whether to using a discrete graphics card or the gpu.",
    )

    # data params that can be changed, default==None means use ones in .json file.
    parser.add_argument("--slen", type=int, default=None)
    parser.add_argument("--max-galaxies", type=int, default=None)
    parser.add_argument("--mean-galaxies", type=int, default=None)

    # training params
    parser.add_argument("--n-images", type=int, default=320)
    parser.add_argument("--n-epochs", type=int, default=201)
    parser.add_argument("--batchsize", type=int, default=32)
    parser.add_argument("--print-every", type=int, default=20)

    parser.add_argument("--ptile-slen", type=int, default=20)
    parser.add_argument("--step", type=int, default=5)
    parser.add_argument("--edge-padding", type=int, default=5)
    parser.add_argument("--max-detections", type=int, default=2)

    args = parser.parse_args()
    main(args)
