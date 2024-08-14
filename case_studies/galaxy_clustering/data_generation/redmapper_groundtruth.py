# flake8: noqa
import os

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS

DES_DIR = (
    "/nfs/turbo/lsa-regier/scratch/gapatron/desdr-server.ncsa.illinois.edu/despublic/dr2_tiles"
)
BOUNDING_BOXES_PATH = "/data/scratch/des/bounding_coordinates.pickle"
DES_SUBDIRS = (d for d in os.listdir(DES_DIR) if d.startswith("DES"))
DES_SVA_PATH = "/data/scratch/des/sva_map.pickle"
REDMAPPER_PATH = "redmapper_sva1-expanded_public_v6.3_members.fits"

REDMAPPER_CATALOG = Table.read(REDMAPPER_PATH).to_pandas()
BOUNDING_BOXES = pd.read_pickle(BOUNDING_BOXES_PATH)
CLUSTER_INDICES = pd.unique(REDMAPPER_CATALOG["ID"])
DES_SVA_TILES = pd.read_pickle(DES_SVA_PATH)
OUTPUT_DIR = "/data/scratch/des/redmapper_groundtruth"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)


def find_containing_tile(cluster_cat):
    for k, v in BOUNDING_BOXES.items():
        if k not in DES_SVA_TILES:
            continue
        ra_max, ra_min, dec_max, dec_min = v["RA_max"], v["RA_min"], v["DEC_max"], v["DEC_min"]
        ra, dec = cluster_cat.iloc[0]["RA"], cluster_cat.iloc[0]["DEC"]
        if (ra_min < ra < ra_max) and (dec_min < dec < dec_max):
            return k
    return None


def find_cluster_location(cluster_cat, containing_tile):
    tile_path = f"{DES_DIR}/{containing_tile}"
    for file in os.listdir(tile_path):
        if file.endswith("_r_nobkg.fits.fz"):
            filename = os.path.join(tile_path, file)
            break
    hdu_list = fits.open(filename)
    ra_max = max(cluster_cat["RA"])
    dec_max = max(cluster_cat["DEC"])
    ra_min = min(cluster_cat["RA"])
    dec_min = min(cluster_cat["DEC"])
    w = WCS(hdu_list[1].header)
    x_min, y_max = w.all_world2pix(ra_max, dec_max, 1)
    x_max, y_min = w.all_world2pix(ra_min, dec_min, 1)
    return int(x_min), int(x_max) + 1, int(y_min), int(y_max) + 1


def process_cluster(cluster_idx, sva_write_flags):
    cluster_cat = REDMAPPER_CATALOG[REDMAPPER_CATALOG["ID"] == cluster_idx]
    containing_tile = find_containing_tile(cluster_cat)
    if containing_tile is None:
        print(f"Could not find tile for cluster idx {cluster_idx}. Moving on ...")
        return None
    x_min, x_max, y_min, y_max = find_cluster_location(cluster_cat, containing_tile)
    output_file = f"{OUTPUT_DIR}/{containing_tile}_redmapper_groundtruth.npy"
    if not sva_write_flags[containing_tile]:
        output_array = np.full((10000, 10000), False)
        output_array[x_min:x_max, y_min:y_max] = True
        np.save(output_file, output_array)
    else:
        output_array = np.load(output_file)
        output_array[x_min:x_max, y_min:y_max] = True
        np.save(output_file, output_array)
        print(f"Rewriting {containing_tile} ...")
    return containing_tile


def main():
    sva_write_flags = dict.fromkeys(DES_SVA_TILES, False)
    files_written_count = 0
    for cluster_idx in CLUSTER_INDICES:
        written_tile = process_cluster(cluster_idx, sva_write_flags)
        if written_tile is not None and not sva_write_flags[written_tile]:
            files_written_count += 1
            print(f"Writing output for file {files_written_count} out of {len(DES_SVA_TILES)}")
            sva_write_flags[written_tile] = True

    for k, v in sva_write_flags.items():
        if not v:
            output_array = np.full((10000, 10000), False)
            output_file = f"{OUTPUT_DIR}/{k}_redmapper_groundtruth.npy"
            np.save(output_file, output_array)
            files_written_count += 1
            print(f"Writing output for file {files_written_count} out of {len(DES_SVA_TILES)}")


if __name__ == "__main__":
    main()
