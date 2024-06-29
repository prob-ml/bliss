import os
import sys

import numpy as np
import pandas as pd
from astropy.io import ascii as astro_ascii
from astropy.table import Table

from case_studies.galaxy_clustering.background_prior import BackgroundPrior
from case_studies.galaxy_clustering.cluster_prior import ClusterPrior

DATA_PATH = "/home/kapnadak/bliss/case_studies/galaxy_clustering/data"
CATALOG_PATH = os.path.join(DATA_PATH, "catalogs")
FILE_PREFIX = "galsim_des"
CLUSTER_PROB = 0.5


def main(**kwargs):
    if not os.path.exists(CATALOG_PATH):
        os.makedirs(CATALOG_PATH)

    cluster_prior = ClusterPrior(
        size=int(kwargs.get("nfiles", 100)), image_size=int(kwargs.get("image_size", 1280))
    )
    background_prior = BackgroundPrior(
        size=int(kwargs.get("nfiles", 100)), image_size=int(kwargs.get("image_size", 1280))
    )

    catalogs, global_catalog = cluster_prior.sample_cluster()
    background_catalogs = background_prior.sample_background()
    combined_catalogs = []
    for i, background_catalog in enumerate(background_catalogs):
        if np.random.uniform() < CLUSTER_PROB:
            combined_catalogs.append(pd.concat([catalogs[i], background_catalog]))
        else:
            combined_catalogs.append(background_catalog)
            global_catalog.drop(i, inplace=True)

    global_filename = f"{DATA_PATH}/global_catalog.dat"
    global_catalog.to_csv(global_filename)
    for i, catalog in enumerate(combined_catalogs):
        file_name = f"{CATALOG_PATH}/{FILE_PREFIX}_{i:03}.dat"
        catalog_table = Table.from_pandas(catalog)
        astro_ascii.write(catalog_table, file_name, format="no_header", overwrite=True)


if __name__ == "__main__":
    main(**dict(arg.split("=") for arg in sys.argv[1:]))
