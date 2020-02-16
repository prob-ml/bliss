#!/usr/bin/env python3
"""
Should be executed from galaxy-net folder.
"""
import sys
from os.path import dirname

sys.path.insert(0, dirname(dirname(__file__)))  # galaxy-net

import subprocess
import argparse
from src.data import generate
from src import utils
from src.visualize import draw_utils
import h5py

# ToDo: add how many images you want.
parser = argparse.ArgumentParser(description='Generate images',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--dir', type=str, default='test')
parser.add_argument('--filename', type=str, default='images')

pargs = parser.parse_args()

# filepath = utils.data_path.joinpath(f"processed/test1/images.hdf5")

generate.generate_images("galcatsim", pargs.dir, pargs.filename, num_images=5,
                         slen=40, num_bands=6, fixed_size=False, sky_factor=50,
                         prop_name=f"prop_{pargs.filename}.txt")

# with h5py.File(filepath, 'r') as f:
#     img = f['img0']
#     img2 = f['img1']
#     draw_utils.draw_multiband(img, filename='img0', figsize=(16, 16))
#     draw_utils.draw_multiband(img2, filename='img1', figsize=(16, 16))
