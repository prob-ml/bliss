import os
import sys

from astropy.io import ascii as astro_ascii
from astropy.table import Table

from case_studies.galaxy_clustering.prior import GalaxyClusterPrior

DATA_PATH = "/data/scratch/kapnadak/data_new"
CATALOG_PATH = os.path.join(DATA_PATH, "catalogs")
FILE_PREFIX = "galsim_des"


def main(**kwargs):
    if not os.path.exists(CATALOG_PATH):
        os.makedirs(CATALOG_PATH)

    cluster_prior_obj = GalaxyClusterPrior(
        size=int(kwargs.get("nfiles", 100)), image_size=int(kwargs.get("image_size", 1280))
    )

    catalogs, global_catalog = cluster_prior_obj.sample()
    global_filename = f"{DATA_PATH}/global_catalog.dat"
    global_catalog.to_csv(global_filename)
    for i, catalog in enumerate(catalogs):
        file_name = f"{CATALOG_PATH}/{FILE_PREFIX}_{i:03}.dat"
        catalog_table = Table.from_pandas(catalog)
        astro_ascii.write(catalog_table, file_name, format="no_header", overwrite=True)


if __name__ == "__main__":
    main(**dict(arg.split("=") for arg in sys.argv[1:]))
