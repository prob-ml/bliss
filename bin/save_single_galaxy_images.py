#!/usr/bin/env python3

import argparse
import h5py
import torch
import multiprocessing

from . import setup_paths, add_path_args

from celeste.datasets import galaxy_datasets

datasets = {"CatsimGalaxies": galaxy_datasets.CatsimGalaxies}

final_image_name = "images.hd5f"
background_name = "background.pt"


def merge_files(output_path):
    new_file_path = output_path.joinpath(final_image_name)
    assert not new_file_path.exists(), "The merged file is already there."

    h5_files = [pth for pth in output_path.iterdir() if pth.suffix == ".hdf5"]

    # first, we figure out the total number of images to copy.
    total_images = 0.0
    for pth in h5_files:
        with h5py.File(pth, "r") as curr_file:
            ds = curr_file["images"]
            total_images += ds.shape[0]
            shape = ds.shape
            dtype = ds.dtype

    # then we copy them.
    with h5py.File(new_file_path, "w") as new_file:
        fds = new_file.create_dataset(
            "images", shape=(total_images, *shape[1:]), dtype=dtype
        )
        images_copied = 0
        for pth in h5_files:
            with h5py.File(pth, "r") as curr_file:
                ds = curr_file["images"]
                num_images = ds.shape[0]
                fds[images_copied : images_copied + num_images, :, :, :] = ds[
                    :, :, :, :
                ]
                fds.attrs["background"] = ds.attrs["background"]
                images_copied += num_images


def save_background(output_path):
    new_file_path = output_path.joinpath(final_image_name)
    background_path = output_path.joinpath(background_name)
    with h5py.File(new_file_path, "r") as curr_file:
        ds = curr_file["images"]
        background = torch.from_numpy(ds.attrs["background"])
        torch.save(background, background_path)


def main(args):
    assert (
        args.n_processes <= multiprocessing.cpu_count()
    ), "Requesting more cpus than available."

    # load paths.
    paths = setup_paths(args)

    prop_file_path = paths["output"].joinpath("prop.txt")
    file_paths = [
        paths["output"].joinpath(f"image{i}.hdf5") for i in range(args.n_processes)
    ]

    # load dataset and save properties
    dataset = datasets[args.dataset].from_args(args)
    with open(prop_file_path, "w") as prop_file:
        dataset.print_props(prop_file)

    # prepare multiprocessing.
    generate_args = [(dataset, fp, args.n_images_per_process) for fp in file_paths]
    with multiprocessing.Pool(processes=args.n_processes) as pool:
        pool.starmap(galaxy_datasets.save_images, generate_args)

    merge_files(paths["output"])
    save_background(paths["output"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Use multiprocessing to save images from the specified dataset into "
        "images.hdf5 file in the default data directory, along with the background used for "
        "this images in background.pt"
    )

    # ---------------
    # Paths
    # ----------------
    parser = add_path_args(parser)

    # ---------------
    # Datasets
    # ----------------
    # properties of individually saved images.
    parser.add_argument("--dataset", type=str, choices=[*datasets], required=True)
    parser.add_argument("--n-images-per-process", type=int, required=True)
    parser.add_argument(
        "--n-processes",
        type=int,
        help="Number of parallel processes to run to write images in disk.",
        required=True,
    )
    parser.add_argument("--slen", type=int, default=51)
    parser.add_argument("--n-bands", type=int, default=1)
    parser.add_argument("--snr", type=float, default=200, help="SNR to use for noise")

    catsim_group = parser.add_argument_group("[Catsim Dataset]")
    galaxy_datasets.CatsimGalaxies.add_args(catsim_group)

    pargs = parser.parse_args()
    assert pargs.n_bands in [1, 6] or not pargs.generate, "Only 1 or 6 bands suggested."

    main(pargs)
