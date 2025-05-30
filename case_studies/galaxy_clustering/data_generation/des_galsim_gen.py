# flake8: noqa
# Utilities
import os
from pathlib import Path
# Generate DES DR1 catalogs for galaxy clustering
from astropy.io import ascii as astro_ascii
from astropy.table import Table
from case_studies.galaxy_clustering.data_generation.prior import Prior
# Render Galsim images for galaxy clustering
from case_studies.galaxy_clustering.data_generation.gen_utils import galsim_render_band
# Generate file data (with tile catalogs and images) for galaxy clustering
from data_gen import file_data_gen
# Configuration management
import hydra



DES_DIR = Path(
    "/nfs/turbo/lsa-regier/scratch/gapatron/desdr-server.ncsa.illinois.edu/despublic/dr1_tiles/"
)
DES_SUBDIRS = [d for d in os.listdir(DES_DIR) if d.startswith("DES")]
OUTPUT_DIR = "/nfs/turbo/lsa-regier/scratch/gapatron/des_synthetic"
PSF_DIR = Path("/nfs/turbo/lsa-regier/scratch/gapatron/psf-models/dr1_tiles")
DES_PIXEL_SCALE = 0.263

IMAGE_SIZE = 10000
NFILES = 1
GALSIM_PATH = (
    "/home/kapnadak/bliss/case_studies/galaxy_clustering"
    "/data_generation/custom-single-image-galsim.yaml"
)
BANDS = ("g", "r", "i", "z")



def catalog_gen(cfg):
    """Generate DES DR1 catalogs for galaxy clustering.
    Amounts to mapping DES DR1 columns to a galsim catalog format.
    It creates a catalog for each DES tile and saves it in the specified output directory.

    Args:
        cfg: configuration object containing parameters for catalog generation

    
    """
    data_path = cfg.data_dir
    des_subdirs = cfg.desdr_subdirs
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    
    prior = Prior(
            image_size=cfg.image_size,
            load_cluster_catalog=True # Catalog is hardcoded in the Prior class for now.
        )
    print(f"Generating catalogs for {len(os.listdir(des_subdirs))} DES tiles...")
    # How to lexicographically sort the subdirs?
    for i, des_subdir in enumerate(sorted(os.listdir(des_subdirs))):
        print(f"Processing tile {des_subdir}...")
        des_catalog = prior.make_des_catalog(
            des_subdir=des_subdir,
            class_star_thr=cfg.class_star_thr,
        )
        catalog_dir = f"{data_path}/catalogs"
        if not os.path.exists(catalog_dir):
            os.makedirs(catalog_dir)

        filename = f"{data_path}/catalogs/{des_subdir}.cat"
        catalog_table = Table.from_pandas(des_catalog)
        astro_ascii.write(catalog_table, filename, format="no_header", overwrite=True)
        print(f"Finished processing tile {i + 1}/{len(os.listdir(des_subdirs))}: {des_subdir}")
    print("Catalog generation completed.")


def image_gen(cfg):
    """Generate Galsim images for galaxy clustering.

    Args:
        cfg: configuration object containing parameters for image generation
    """
    data_path = cfg.data_dir
    if not os.path.exists(data_path):
        os.makedirs(data_path)

    des_subdirs = sorted(os.listdir(DES_DIR))
    for i, des_subdir in enumerate(des_subdirs):
        for band in BANDS:
            # DEFINE THINGS HERE
            os.makedirs(f"{data_path}/images/{des_subdir}", exist_ok=True)
            print(f"Processing tile {des_subdir} for band {band}...")
            galsim_render_band(band=band,
                               des_subdir=des_subdir,
                               data_path=cfg.data_dir,
                               galsim_confpath=cfg.galsim_confpath,
                               psf_filepath=cfg.psf_model_path
                               )
        print(f"Finished processing tile {i + 1}/{len(des_subdirs)}")



@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg):
    #catalog_gen(cfg.data_gen)
    image_gen(cfg.data_gen)
    #print("Starting file datum generation process ...")
    #file_data_gen(cfg.data_gen)
    #print(" ... Done!")


if __name__ == "__main__":
    main()
