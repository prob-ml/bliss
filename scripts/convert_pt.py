import os
from pathlib import Path

import torch

from bliss.catalog import TileCatalog

cached_data_path = "/data/scratch/regier/m2"
out_path = "/data/scratch/regier/m2_full_cat"

tile_slen = 4

os.makedirs(out_path, exist_ok=True)

for fname in os.listdir(cached_data_path):
    if not fname.endswith(".pt"):
        continue
    print(f"processing {fname}")
    datums_in = torch.load(Path(cached_data_path, fname))
    datums_out = []

    for x in datums_in:
        y = {}
        for k, v in x.items():
            if k == "tile_catalog":
                d = {k2: v2.unsqueeze(0) for k2, v2 in v.items()}
                if "galaxy_params" in d:
                    del d["galaxy_params"]
                if "galaxy_fluxes" in d:
                    del d["galaxy_fluxes"]
                tile_cat = TileCatalog(tile_slen, d)
                y["full_catalog"] = tile_cat.to_full_catalog().data
            elif k == "deconvolution":
                continue
            else:
                y[k] = v
        datums_out.append(y)
    torch.save(datums_out, Path(out_path, fname))
