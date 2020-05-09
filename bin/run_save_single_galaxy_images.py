#!/usr/bin/env python3

import argparse
import h5py
import numpy as np
import multiprocessing

from celeste.utils import const
from celeste.datasets import galaxy_datasets
from celeste.datasets.galaxy_datasets import generate_images

all_datasets = {
    cls.__name__: cls for cls in galaxy_datasets.GalaxyDataset.__subclasses__()
}

final_image_name = "images.hd5f"
background_name = "background.npy"


def generate(
    n_images, file_path, output_path, dataset, overwrite=False,
):
    """
    Generate a single image .hdf5 file with the dataset and dataset parameters specified.
    """

    prop_file_path = output_path.joinpath(f"prop_{file_path.name}.txt")

    assert (
        not file_path.exists() or overwrite
    ), "Trying to overwrite file without --overwrite"

    generate_images(
        dataset, file_path, prop_file_path=prop_file_path, n_images=n_images
    )


def merge_files(output_path):
    new_file_path = output_path.joinpath(final_image_name)
    assert not new_file_path.exists(), "The merged file is already there."

    h5_files = [pth for pth in output_path.iterdir() if pth.suffix == ".hdf5"]

    # first, we figure out the total number of images to copy.
    total_images = 0.0
    for pth in h5_files:
        with h5py.File(pth, "r") as curr_file:
            ds = curr_file[const.image_h5_name]
            total_images += ds.shape[0]
            shape = ds.shape
            dtype = ds.dtype

    # then we copy them.
    with h5py.File(new_file_path, "w") as new_file:
        fds = new_file.create_dataset(
            const.image_h5_name, shape=(total_images, *shape[1:]), dtype=dtype
        )
        images_copied = 0
        for pth in h5_files:
            with h5py.File(pth, "r") as curr_file:
                ds = curr_file[const.image_h5_name]
                num_images = ds.shape[0]
                fds[images_copied : images_copied + num_images, :, :, :] = ds[
                    :, :, :, :
                ]
                fds.attrs[const.background_h5_name] = ds.attrs[const.background_h5_name]
                images_copied += num_images


def save_background(output_path):
    new_file_path = output_path.joinpath(final_image_name)
    background_path = output_path.joinpath(background_name)
    with h5py.File(new_file_path, "r") as curr_file:
        ds = curr_file[const.image_h5_name]
        background = ds.attrs[const.background_h5_name]
        np.save(background_path, background)


def main(args):

    assert (
        args.n_processes <= multiprocessing.cpu_count()
    ), "Requesting more cpus than available."
    output_path = const.data_path.joinpath(f"{args.out_dir}")
    output_path.mkdir(exist_ok=True)

    # load dataset only once.
    dataset = all_datasets[args.dataset_name].load_dataset_from_params(
        args.dset_param_file
    )

    file_paths = [output_path.joinpath(f"image{i}") for i in range(args.n_processes)]
    generate_args = [
        (args.n_images_per_process, fp, output_path, dataset, args.overwrite)
        for fp in file_paths
    ]
    with multiprocessing.Pool(processes=args.n_processes) as pool:
        pool.starmap(generate_images, generate_args)

    merge_files(output_path)
    save_background(output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Use multiprocessing to save images from the specified dataset into .hdf5 file in the default "
        "data directory, along with the background used for this images in background.npy"
    )

    # io stuff
    parser.add_argument(
        "--out-dir",
        type=str,
        required=True,
        description="Will write results in data/{output_dir}.",
    )
    parser.add_argument(
        "--filename",
        type=str,
        default="images",
        description="name for final image file.",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Whether to overwrite existing files"
    )

    # properties of individually saved images.
    parser.add_argument("--dataset", type=str, choices=all_datasets.keys())
    parser.add_argument(
        "--dset-params-file",
        type=str,
        default=None,
        help="Specify params file relative to data folder, "
        "from which to load params for dataset.",
    )
    parser.add_argument("--images-per-process", type=int, required=True)
    parser.add_argument("--slen", type=int, default=51)
    parser.add_argument("--n-bands", type=int)

    # multiprocessing
    parser.add_argument(
        "--n-processes",
        type=str,
        help="Number of parallel processes " "to run to write images in disk.",
        required=True,
    )

    pargs = parser.parse_args()

    assert (
        pargs.num_bands in [1, 6] or not pargs.generate
    ), "Only 1 or 6 bands supported."

    main(pargs)
