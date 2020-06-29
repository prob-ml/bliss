#!/usr/bin/env python
import argparse
import os

from . import setup_paths
from . import setup_device

from celeste import train
from celeste.models import encoder, decoder


# TODO: part of this function can probably be a more general utility function in
#       simulated_datasets.py


def main(args):

    paths = setup_paths(args)
    device = setup_device(args)

    out_dir = paths["results"].joinpath(args.output_name) if args.output_name else None

    print(
        f"running sleep phase for n_epochs={args.n_epochs}, batch_size={args.batch_size}, "
        f"n_images={args.n_images}, device={device}"
    )
    print(f"output dir: {out_dir}")

    galaxy_dataset = setup_dataset(args, paths)

    galaxy_encoder = encoder.ImageEncoder(
        slen=args.slen,
        n_bands=args.n_bands,
        ptile_slen=args.ptile_slen,
        step=args.step,
        edge_padding=args.edge_padding,
        max_detections=args.max_detections,
        n_source_params=galaxy_dataset.simulator.latent_dim,
    ).to(device)

    train_sleep = train.SleepTraining(
        galaxy_encoder,
        galaxy_dataset,
        args.slen,
        n_bands=1,
        n_source_params=galaxy_dataset.simulator.latent_dim,
        batch_size=args.batch_size,
        eval_every=args.eval_every,
        out_dir=out_dir,
        verbose=True,
        torch_seed=args.torch_seed,
        np_seed=args.np_seed,
    )

    print("training starting...")
    train_sleep.run(args.n_epochs)


# TODO: add more command line arguments corresponding to star simulation (like f_min, f_max,...)
#       right now all are default.
if __name__ == "__main__":

    # Setup arguments.
    parser = argparse.ArgumentParser(
        description="Sleep phase galaxy training [argument parser]",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--no-cuda",
        action="store_true",
        help="whether to using a discrete graphics card or not.",
    )

    # data params that can be changed, default==None means use ones in .json file.

    parser.add_argument("--max-sources", type=int, default=10)
    parser.add_argument("--mean-sources", type=int, default=5)
    parser.add_argument("--min-sources", type=int, default=1)

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

    parser.add_argument(
        "--galaxy-decoder-file",
        type=str,
        default="decoder_params_100_single_band_i.dat",
        help="File relative to data directory containing galaxy decoder state_dict.",
    )

    parser.add_argument(
        "--background-file",
        type=str,
        default="background_galaxy_single_band_i.npy",
        help="File relative to data directory containing background to be used.",
    )

    parser.add_argument(
        "--psf-file",
        type=str,
        default="fitted_powerlaw_psf_params.npy",
        help="File relative to data directory containing PSF to be used.",
    )

    pargs = parser.parse_args()
    main(pargs)
