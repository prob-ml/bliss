#!/usr/bin/env python

import argparse

from celeste import utils
from celeste import train
from celeste.models import sourcenet_lib
from celeste.datasets import simulated_datasets


def main(args):
    print(
        f"running sleep phase for n_epochs={args.n_epochs}, batchsize={args.batchsize}, n_images={args.n_images}"
    )

    utils.set_device(args.device, args.no_cuda)  # set global device to use.

    data_params = utils.load_data_params_from_args(
        "dataset_params/default_galaxy_parameters.json", args
    )

    print("data params to be used:", data_params)

    galaxy_dataset = simulated_datasets.GalaxyDataset.load_dataset_from_params(
        args.n_images, data_params
    )

    galaxy_encoder = sourcenet_lib.SourceEncoder(
        slen=data_params["slen"],
        n_bands=data_params["n_bands"],
        ptile_slen=args.ptile_slen,
        step=args.step,
        edge_padding=args.edge_padding,
        max_detections=args.max_detections,
        n_source_params=galaxy_dataset.simulator.latent_dim,
    ).to(utils.device)

    train_sleep = train.SleepTraining(
        galaxy_encoder,
        galaxy_dataset,
        data_params["slen"],
        num_bands=1,
        n_source_params=galaxy_dataset.simulator.latent_dim,
        batchsize=args.batchsize,
        eval_every=args.print_every,
        out_name=pargs.results_dir,
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
        "--results-dir",
        type=str,
        default="test",
        metavar="DIR",
        help="Directory in results path, where output will be saved.",
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
