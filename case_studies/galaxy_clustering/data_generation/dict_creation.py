from pathlib import Path
import os
from astropy.io import fits
import pandas as pd
import pickle

def main():

    n_tiles = 1
    DES_DIR = Path(
        "/nfs/turbo/lsa-regier/scratch/gapatron/desdr-server.ncsa.illinois.edu/despublic/dr2_tiles/"
    )
    DES_SUBDIRS = [d for d in os.listdir(DES_DIR) if d.startswith('DES')]
    obj_tile_mapping = {}
    for DES_SUBDIR in DES_SUBDIRS:
        catalog_path = DES_DIR / Path(DES_SUBDIR) / Path(f"{DES_SUBDIR}_dr2_main.fits")
        # bad index for now
        if DES_SUBDIR == 'DES2359-4540':
            catalog_path = "/home/kapnadak/bliss/case_studies/galaxy_clustering/DES2359-4540_dr2_main.fits"
        print(f"Processing tile {n_tiles} ...")
        catalog_data = fits.getdata(catalog_path)
        source_df = pd.DataFrame(catalog_data)
        for obj_id in source_df["COADD_OBJECT_ID"]:
            obj_tile_mapping[obj_id] = DES_SUBDIR
        n_tiles += 1
    
    with open('obj_to_tile.pickle', 'wb') as handle:
        pickle.dump(obj_tile_mapping, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()