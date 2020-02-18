#!/usr/bin/env python3
"""
Should be executed from galaxy-net folder.
"""
import sys
from os.path import dirname

sys.path.insert(0, dirname(dirname(__file__)))  # galaxy-net

import argparse
from src.data import generate
from src import utils
import h5py

# ToDo: add how many images you want.
parser = argparse.ArgumentParser(description='Generate images',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--dir', type=str, required=True)
parser.add_argument('--filename', type=str, default=None)
parser.add_argument('--num-images', type=int, default=None)
parser.add_argument('--overwrite', action='store_true')
parser.add_argument('--generate', action='store_true')
parser.add_argument('--merge', action='store_true', help="Merge all files from directory with .hdf5 extension.")

pargs = parser.parse_args()

assert pargs.generate or pargs.merge, "At least one of generate/merge is required."

# files and directories needed
output_path = utils.data_path.joinpath(f"processed/{pargs.dir}")
output_path.mkdir(exist_ok=True)

if pargs.generate:
    assert pargs.filename is not None and pargs.num_images is not None, "Required arguments for generate."
    prop_file_path = output_path.joinpath(f"prop_{pargs.filename}.txt")
    out_path = output_path.joinpath(f"{pargs.filename}.hdf5")

    assert not out_path.exists() or pargs.overwrite, "Trying to overwrite file."

    generate.generate_images("galcatsim", out_path, prop_file_path, num_images=pargs.num_images,
                             slen=40, num_bands=6, fixed_size=False, sky_factor=50)

if pargs.merge:
    new_file_path = output_path.joinpath("images.hdf5")

    total_images = 0.
    for pth in output_path.iterdir():
        if pth.suffix == ".hdf5":
            with h5py.File(pth, 'r') as currfile:
                ds = currfile[utils.image_h5_name]
                total_images += ds.shape[0]
                shape = ds.shape
                dtype = ds.dtype

    with h5py.File(new_file_path, 'w') as nfile:
        fds = nfile.create_dataset(utils.image_h5_name, shape=(total_images, *shape[1:]), dtype=dtype)
        for i, pth in enumerate(output_path.iterdir()):
            if pth.suffix == ".hdf5":
                with h5py.File(pth, 'r') as currfile:
                    ds = currfile[utils.image_h5_name]
                    num_images = ds.shape[0]
                    fds[i * num_images:(i + 1) * num_images, :, :, :] = ds[:, :, :, :]
                    fds.attrs[utils.background_h5_name] = ds.attrs[utils.background_h5_name]
