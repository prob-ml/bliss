import os
import sys

from astropy.io import ascii as astro_ascii
from astropy.table import Table
from prior import ClusterPrior

ENVIRONMENT_PATH = os.getcwd()
CATALOG_PATH = os.path.join(ENVIRONMENT_PATH, "data/catalogs")
PADDED_CATALOG_PATH = os.path.join(ENVIRONMENT_PATH, "data/padded_catalogs")
FILE_PREFIX = "galsim_des"
MAX_SOURCES = 2200
N_FILES = sys.argv[1]
if not os.path.exists(CATALOG_PATH):
    os.makedirs(CATALOG_PATH)
if not os.path.exists(PADDED_CATALOG_PATH):
    os.makedirs(PADDED_CATALOG_PATH)

if N_FILES:
    cluster_prior_obj = ClusterPrior(size=int(N_FILES))
else:
    cluster_prior_obj = ClusterPrior()
catalogs = cluster_prior_obj.sample()
for i, catalog in enumerate(catalogs):
    file_name = f"data/catalogs/{FILE_PREFIX}_{i:03}.dat"
    catalog_table = Table.from_pandas(catalog)
    astro_ascii.write(catalog_table, file_name, format="no_header", overwrite=True)
    padded_catalog = catalog.reindex(range(MAX_SOURCES), fill_value=0)
    padded_file_name = f"data/padded_catalogs/{FILE_PREFIX}_padded_{i:03}.dat"
    padded_catalog_table = Table.from_pandas(padded_catalog)
    astro_ascii.write(padded_catalog_table, padded_file_name, format="no_header", overwrite=True)
