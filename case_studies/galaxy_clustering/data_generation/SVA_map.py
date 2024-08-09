# flake8: noqa
import pickle

import numpy as np
import pandas as pd
from astropy.table import Table

SVA_PATH = "/data/scratch/des/sva1_gold_r1.0_catalog.fits"
BOUNDING_BOX_PATH = "/data/scratch/des/bounding_coordinates.pickle"


def main():
    sva_catalog = Table.read(SVA_PATH).to_pandas()
    bounding_boxes = pd.read_pickle(BOUNDING_BOX_PATH)
    des_sva_intersection = []
    output_filename = "/data/scratch/des/sva_map.pickle"

    for k, v in bounding_boxes.items():
        ra_min, ra_max, dec_min, dec_max = v["RA_min"], v["RA_max"], v["DEC_min"], v["DEC_max"]
        ra_intersection = np.logical_and((ra_min < sva_catalog["RA"]), (sva_catalog["RA"] < ra_max))
        dec_intersection = np.logical_and(
            (dec_min < sva_catalog["DEC"]), (sva_catalog["DEC"] < dec_max)
        )
        full_intersection = np.logical_and(ra_intersection, dec_intersection)
        if full_intersection.any():
            des_sva_intersection.append(k)

    with open(output_filename, "wb") as handle:
        pickle.dump(des_sva_intersection, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()
