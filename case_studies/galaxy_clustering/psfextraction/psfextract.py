import argparse
from astropy.io import fits
import numpy as np
import os
import subprocess

TILE_PATHS = '/nfs/turbo/lsa-regier/scratch/gapatron/desdr-server.ncsa.illinois.edu/despublic/dr1_tiles'
DES_BANDS = ("g", "r", "i", "z")
SAVE_PATH = '/data/scratch/gapatron/galaxy_clustering/desdr1_galsim/psf_models/dr1_tiles'
CONFIG_PATH = '/data/scratch/gapatron/galaxy_clustering/desdr1_galsim/config'
SEED = 42


def decompress_hdu(filepath, savepath):
    """
    Decompress HDU for SExtractor processing.
    """
    with fits.open(filepath) as hdul:
        decompressed_hdul = fits.HDUList()
        for (i,hdu) in enumerate(hdul):
            if isinstance(hdu, fits.CompImageHDU):
                #Only save HDU 0 and HDU 1
                if i != 1:
                    continue
                decompressed_data = hdu.data  
                decompressed_header = hdu.header
                new_hdu = fits.ImageHDU(data=decompressed_data, header=decompressed_header)
                decompressed_hdul.append(new_hdu)
                
            else:
                decompressed_hdul.append(hdu)
        decompressed_hdul.writeto(savepath, overwrite=True)




if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--tile_paths',type=str, nargs='+', required=False)
    parser.add_argument('--seed',type=int, required=False)
    args = parser.parse_args()

    tile_paths = args.tile_paths if not args.tile_paths is None else os.listdir(TILE_PATHS)
    
    print(f"The following tiles will be modeled: {tile_paths}")

    for tile in sorted(tile_paths):
        ## Define directories where data will be stored
        save_dir = f"{SAVE_PATH}/{tile}"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        print(f"Processing tile: {tile}")
        print("########################################")
        print(tile)
        print("########################################")
        try:
            dir_files = {
                    band: [
                        f for f in os.listdir(f"{TILE_PATHS}/{tile}") if f.endswith(f"_{band}.fits.fz")
                    ][0]
                    for band in DES_BANDS
                }
        except:
            continue
        
        for band, filename in dir_files.items():
            # Where you read it from, in gapatron desdr dir
            filepath = f"{TILE_PATHS}/{tile}/{filename}"
            # Where you save it in gapatron psf-models/dir
            savepath = f"{SAVE_PATH}/{tile}/{filename}".replace(".fits.fz", ".fits")

            if os.path.exists(savepath.replace(".fits.fz", ".psf")):
                print(f"PSF already exists for {savepath}. Skipping...")
                continue

            print(f"Decompressing: {filepath}")
            decompress_hdu(filepath, savepath)
            print(f"Success! Saved at: {savepath}")

            savecatpath = savepath.replace(".fits", ".cat")
            print(f"Creating catalog: {savecatpath}")
            subprocess.run(["sed", "-i", f"s|^CATALOG_NAME\s*.*|CATALOG_NAME     {savecatpath}|", f"{CONFIG_PATH}/config.sex"])
            print(f"Running SExtractor on {savepath} ...")
            subprocess.run(["source-extractor", savepath, "-c", f"{CONFIG_PATH}/config.sex"])
            print(f"Running PSFEx on {savecatpath} ...")
            subprocess.run(["psfex", f"{savecatpath}", "-c", f"{CONFIG_PATH}/config.psfex"])
            subprocess.run(["rm", f"{savepath}"])