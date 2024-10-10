import argparse
from astropy.io import fits
import numpy as np
import os
import subprocess

TILE_PATHS = '/nfs/turbo/lsa-regier/scratch/gapatron/desdr-server.ncsa.illinois.edu/despublic/dr2_tiles'
DES_BANDS = ("g", "r", "i", "z")
SAVE_PATH = '/nfs/turbo/lsa-regier/scratch/gapatron/psf-models/dr2_tiles'
CONFIG_PATH = '/nfs/turbo/lsa-regier/scratch/gapatron/psf-models/config'
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



def sample_tiles(path_to_tiles, n_samples_per_band=100, seed=42):
    """
    Sample n_samples_per_band tiles to model with SExtractor.
    """
    np.random.seed(seed)
    samples = np.random.choice(os.listdir(path_to_tiles), size=n_samples_per_band)
    return samples



if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_samples_per_band',type=int, required=True)
    parser.add_argument('--seed',type=int, required=False)
    args = parser.parse_args()

    if not args.seed:
        args.seed = SEED

    tile_paths = sample_tiles(
                        TILE_PATHS,
                        n_samples_per_band=args.n_samples_per_band,
                        seed=args.seed
                        )
    
    print(f"The following tiles will be modeled: {tile_paths}")

    for tile in tile_paths:
        ## Define directories where data will be stored
        save_dir = f"{SAVE_PATH}/{tile}"
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        dir_files = {
                band: [
                    f for f in os.listdir(f"{TILE_PATHS}/{tile}") if f.endswith(f"{band}_nobkg.fits.fz")
                ][0]
                for band in DES_BANDS
            }
        
        for band, filename in dir_files.items():
            # Where you read it from, in gapatron desdr dir
            filepath = f"{TILE_PATHS}/{tile}/{filename}"
            # Where you save it in gapatron psf-models/dir
            savepath = f"{SAVE_PATH}/{tile}/{filename}".replace(".fits.fz", ".fits")

            print(f"Decompressing: {filepath}")
            decompress_hdu(filepath, savepath)
            print(f"Success! Saved at: {savepath}")

            savecatpath = savepath.replace(".fits", ".cat")

            subprocess.run(["sed", "-i", f"s|^CATALOG_NAME\s*.*|CATALOG_NAME     {savecatpath}|", f"{CONFIG_PATH}/config.sex"])
            subprocess.run(["source-extractor", savepath, "-c", f"{CONFIG_PATH}/config.sex"])
            subprocess.run(["psfex", f"{savecatpath}", "-c", f"{CONFIG_PATH}/config.psfex"])