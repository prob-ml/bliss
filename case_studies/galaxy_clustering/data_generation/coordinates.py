import os
import pickle
from pathlib import Path

import pandas as pd
from astropy.io import fits


def main():
    des_dir = Path(
        "/nfs/turbo/lsa-regier/scratch/gapatron/desdr-server.ncsa.illinois.edu/despublic/dr2_tiles/"
    )
    des_subdirs = [d for d in os.listdir(des_dir) if d.startswith("DES")]
    bounding_coordinates = {}
    output_filename = "/data/scratch/des/bounding_coordinates.pickle"
    for des_subdir in des_subdirs:
        catalog_path = des_dir / Path(des_subdir) / Path(f"{des_subdir}_dr2_main.fits")
        catalog_data = fits.getdata(catalog_path)
        source_df = pd.DataFrame(catalog_data)
        ra_min, ra_max = source_df["RA"].min(), source_df["RA"].max()
        dec_min, dec_max = source_df["DEC"].min(), source_df["DEC"].max()
        bounding_box = {
            "RA_min": ra_min,
            "RA_max": ra_max,
            "DEC_min": dec_min,
            "DEC_max": dec_max,
        }
        bounding_coordinates[des_subdir] = bounding_box

    with open(output_filename, "wb") as handle:
        pickle.dump(bounding_coordinates, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()
