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
    obj_tile_mapping = {}
    output_filename = "/data/scratch/des/obj_to_tile.pickle"
    for des_subdir in des_subdirs:
        catalog_path = des_dir / Path(des_subdir) / Path(f"{des_subdir}_dr2_main.fits")
        catalog_data = fits.getdata(catalog_path)
        source_df = pd.DataFrame(catalog_data)
        for obj_id in source_df["COADD_OBJECT_ID"]:
            obj_tile_mapping[obj_id] = des_subdir

    with open(output_filename, "wb") as handle:
        pickle.dump(obj_tile_mapping, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()
