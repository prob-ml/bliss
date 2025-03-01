# flake8: noqa
import os
from pathlib import Path

import pandas as pd

from case_studies.galaxy_clustering.utils.image_gen_utils import create_des_catalog, galsim_render

DES_SVA_TILES = pd.read_pickle("/data/scratch/des/sva_map.pickle")
DES_DIR = Path(
    "/nfs/turbo/lsa-regier/scratch/gapatron/desdr-server.ncsa.illinois.edu/despublic/dr2_tiles/"
)
DES_SUBDIRS = [d for d in os.listdir(DES_DIR) if d.startswith("DES")]
OUTPUT_DIR = "/nfs/turbo/lsa-regier/scratch/gapatron/des_synthetic"


def process_tile(des_subdir, output_dir, index):
    """Process a single tile by creating catalog and images.

    Args:
        des_subdir: directory to be processed
        output_dir: output directory
        index: index of current directory
    """
    print(f"Processing tile {index} of {len(DES_SUBDIRS)} ...")
    print("Creating catalog ...")
    create_des_catalog(des_subdir=des_subdir, data_path=output_dir, file_suffix=des_subdir)
    bands = ["g", "r", "i", "z"]
    for band in bands:
        print(f"Creating band {band} ...")
        galsim_render(des_subdir=des_subdir, data_path=output_dir, band=band)


def main():
    for i, des_subdir in enumerate(DES_SUBDIRS):
        process_tile(des_subdir, OUTPUT_DIR, index=i)


if __name__ == "__main__":
    main()
