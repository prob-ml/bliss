# flake8: noqa
# pylint: skip-file
import pickle

import pandas as pd
import torch

if __name__ == "__main__":
    with open("./merged_catalog/merged_catalog_psf_100000.pkl", "rb") as f1:
        p1 = pd.read_pickle(f1)

    with open("./merged_catalog/merged_catalog_with_flux_over_100000_new.pkl", "rb") as f2:
        p2 = pd.read_pickle(f2)

    for k1, v1 in p1.items():
        v2 = p2[k1].values
        v1 = v1.values

        v1 = torch.from_numpy(v1)
        v2 = torch.from_numpy(v2).to(dtype=v1.dtype)
        if not torch.allclose(v1, v2, equal_nan=True):
            print(f"{k1}: different")
        else:
            print(f"{k1} is equal")
