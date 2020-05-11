#!/usr/bin/env python

import argparse


from celeste import sleep, utils
from celeste.models import sourcenet_lib
from celeste.datasets import simulated_datasets


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

    sleep_phase = sleep.GalaxySleep(
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

    utils.set_device(pargs.device, pargs.no_cuda)  # set global device to use.

    set_seed(pargs.seed)
    data_params = load_data_params(pargs)

    state_dict_file, output_file = prepare_filepaths(pargs.results_dir)

    # setup dataset.
    galaxy_dataset = simulated_datasets.GalaxyDataset.load_dataset_from_params(
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
    ).to(utils.device)

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
    parser.add_argument(
        "--n-images", type=int, default=320, help="Number of images in epoch"
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

    args = parser.parse_args()
    main(args)
