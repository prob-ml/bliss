import os
import sys

from astropy.io import ascii as astro_ascii
from astropy.table import Table

from case_studies.galaxy_clustering.prior import GalaxyClusterPrior

ENVIRONMENT_PATH = os.getcwd()
CATALOG_PATH = os.path.join(ENVIRONMENT_PATH, "data/catalogs")
FILE_PREFIX = "galsim_des"

def main(**kwargs):
    if not os.path.exists(CATALOG_PATH):
        os.makedirs(CATALOG_PATH)

    if "nfiles" in kwargs.keys():
        if "image_size" in kwargs.keys():  
            cluster_prior_obj = GalaxyClusterPrior(size=int(kwargs["nfiles"]), image_size=int(kwargs["image_size"]))
        else:
            cluster_prior_obj = GalaxyClusterPrior(size=int(kwargs["nfiles"]))
    else:
        if "image_size" in kwargs.keys():  
            cluster_prior_obj = GalaxyClusterPrior(image_size=int(kwargs["image_size"]))
        else:
            cluster_prior_obj = GalaxyClusterPrior()

    catalogs = cluster_prior_obj.sample()
    for i, catalog in enumerate(catalogs):
        file_name = f"data/catalogs/{FILE_PREFIX}_{i:03}.dat"
        catalog_table = Table.from_pandas(catalog)
        astro_ascii.write(catalog_table, file_name, format="no_header", overwrite=True)

if __name__=='__main__':
    main(**dict(arg.split('=') for arg in sys.argv[1:]))