import os
import sys

import numpy as np
import pandas as pd
from astropy.io import ascii as astro_ascii
from astropy.table import Table

from case_studies.galaxy_clustering.data_generation.prior import BackgroundPrior, ClusterPrior

DATA_PATH = "/nfs/turbo/lsa-regier/scratch/kapnadak/new_data"
CATALOG_PATH = os.path.join(DATA_PATH, "catalogs")
FILE_PREFIX = "galsim_des"
CLUSTER_PROB = 0.5


def main(**kwargs):
    if not os.path.exists(CATALOG_PATH):
        os.makedirs(CATALOG_PATH)

    nfiles = int(kwargs.get("nfiles", 100))
    cluster_prior = ClusterPrior(image_size=int(kwargs.get("image_size", 1280)))
    background_prior = BackgroundPrior(image_size=int(kwargs.get("image_size", 1280)))

    combined_catalogs = []
    for _ in range(nfiles):
        background_catalog = background_prior.sample_background()
        if np.random.uniform() < CLUSTER_PROB:
            cluster_catalog = cluster_prior.sample_cluster()
            combined_catalogs.append(pd.concat([cluster_catalog, background_catalog]))
        else:
            combined_catalogs.append(background_catalog)

    for i, catalog in enumerate(combined_catalogs):
        file_name = f"{CATALOG_PATH}/{FILE_PREFIX}_{i:03}.dat"
        catalog_table = Table.from_pandas(catalog)
        astro_ascii.write(catalog_table, file_name, format="no_header", overwrite=True)


if __name__ == "__main__":
    main(**dict(arg.split("=") for arg in sys.argv[1:]))
