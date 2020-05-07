#!/usr/bin/env python3

import argparse
import h5py
import numpy as np
import subprocess


from celeste.datasets.galaxy_datasets import generate_images
from celeste.utils import const

# ToDo: Make sure that we want slen=50 as the default
parser = argparse.ArgumentParser(
    description="Generate single galaxy images from one of the galaxy datasets and save them to an h5py file."
)

parser.add_argument("--dir", type=str, required=True)
parser.add_argument("--filename", type=str, default=None)
parser.add_argument("--num-images", type=int, default=None)
parser.add_argument("--overwrite", action="store_true")
parser.add_argument("--slen", type=int, default=50)
parser.add_argument("--num-bands", type=int)
parser.add_argument("--generate", action="store_true")
parser.add_argument(
    "--merge",
    action="store_true",
    help="Merge all files from directory with .hdf5 extension.",
)
parser.add_argument(
    "--background",
    action="store_true",
    help="Save background from .h5py merged file into a .npy, which is separate.",
)

pargs = parser.parse_args()


assert (
    pargs.generate or pargs.merge or pargs.background
), "At least one of generate/merge/background is required."
assert pargs.num_bands in [1, 6] or not pargs.generate, "Only 1 or 6 bands supported."

# files and directories needed
output_path = const.data_path.joinpath(f"{pargs.dir}")
output_path.mkdir(exist_ok=True)

if pargs.generate:
    assert (
        pargs.filename is not None and pargs.num_images is not None
    ), "Required arguments for generate."
    prop_file_path = output_path.joinpath(f"prop_{pargs.filename}.txt")
    out_path = output_path.joinpath(f"{pargs.filename}.hdf5")

    assert not out_path.exists() or pargs.overwrite, "Trying to overwrite file."

    generate_images(
        "galcatsim",
        out_path,
        prop_file_path=prop_file_path,
        num_images=pargs.num_images,
        slen=pargs.slen,
        num_bands=pargs.num_bands,
        fixed_size=False,
    )

if pargs.merge:
    new_file_path = output_path.joinpath("images.hdf5")
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
    with h5py.File(new_file_path, "w") as nfile:
        fds = nfile.create_dataset(
            const.image_h5_name, shape=(total_images, *shape[1:]), dtype=dtype
        )
        images_copied = 0
        for pth in h5_files:
            with h5py.File(pth, "r") as currfile:
                ds = currfile[const.image_h5_name]
                num_images = ds.shape[0]
                fds[images_copied : images_copied + num_images, :, :, :] = ds[
                    :, :, :, :
                ]
                fds.attrs[const.background_h5_name] = ds.attrs[const.background_h5_name]
                images_copied += num_images

if pargs.background:
    new_file_path = output_path.joinpath("images.hdf5")
    background_path = output_path.joinpath("background.npy")
    with h5py.File(new_file_path, "r") as curr_file:
        ds = curr_file[const.image_h5_name]
        background = ds.attrs[const.background_h5_name]
        np.save(background_path, background)


def main():
    # spawn multi-processes to generate images.
    num_copies = 10
    num_images = 1000
    dir_name = "test2"
    num_bands = 1
    slen = 51
    ps = []

    for i in range(num_copies):
        ps.append(
            subprocess.Popen(
                f"cd {root_path};"
                f"python -m gmodel.generate_images --dir {dir_name} --filename images{i} "
                f"--num-images {num_images} --overwrite --slen {slen} --generate --num-bands {num_bands}",
                shell=True,
            )
        )

    # wait for all processes to complete.
    _ = [p.communicate() for p in ps]

    # then we merge the results.
    subprocess.run(
        f"cd {root_path};" f"python -m gmodel.generate_images --dir {dir_name} --merge",
        shell=True,
    )


if __name__ == "__main__":
    main()
