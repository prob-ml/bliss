#!/usr/bin/env python3
"""
Should be executed from galaxy-net folder.
"""
import sys
from os.path import dirname

sys.path.insert(0, dirname(dirname(__file__)))  # galaxy-net

import subprocess
from src.data import generate
from src import utils
from src.visualize import draw_utils
import h5py

filepath = utils.data_path.joinpath(f"processed/test1/images.hdf5")

# generate.generate_images("galcatsim", "test1", "images", num_images=5,
#                     slen=40, num_bands=6, fixed_size=False,
#                     image_name='img{}', prop_name="prop.txt")

with h5py.File(filepath, 'r') as f:
    img = f['img0']
    img2 = f['img1']
    draw_utils.draw_multiband(img, filename='img0', figsize=(16, 16))
    draw_utils.draw_multiband(img2, filename='img1', figsize=(16, 16))
