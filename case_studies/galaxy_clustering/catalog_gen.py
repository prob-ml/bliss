import os

from astropy.io import ascii as astro_ascii
from astropy.table import Table
from prior import ClusterPrior

ENVIRONMENT_PATH = os.getcwd()
CATALOG_PATH = os.path.join(ENVIRONMENT_PATH, "data/catalogs")

if not os.path.exists(CATALOG_PATH):
    os.makedirs(CATALOG_PATH)

cluster_prior_obj = ClusterPrior()
catalogs = cluster_prior_obj.sample()
for i, catalog in enumerate(catalogs):
    file_name = f"data/catalogs/catalog_{i:03}.dat"
    catalog_table = Table.from_pandas(catalog)
    astro_ascii.write(catalog_table, file_name, format="no_header", overwrite=True)
