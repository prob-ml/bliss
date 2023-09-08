import argparse
import warnings
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from bliss.catalog import TileCatalog
from bliss.generate import itemize_data
from case_studies.adaptive_tiling.region_catalog import tile_cat_to_region_cat

parser = argparse.ArgumentParser()
parser.add_argument("input_path", type=str, help="Path to directory containing tile-based data")
parser.add_argument("output_path", type=str, help="Path to directory to store converted data")
h = "Overlap to use for conversion. Default is 0.4."
parser.add_argument("--overlap", required=False, type=float, default=0.4, help=h)
args = parser.parse_args()

OVERLAP_SLEN = args.overlap
INPUT_PATH = Path(args.input_path)
OUTPUT_PATH = Path(args.output_path)

if not OUTPUT_PATH.exists():
    Path.mkdir(OUTPUT_PATH, parents=True)

warnings.filterwarnings("error")  # We want the warning to throw an error to know to skip an image

# Iterate over each data file in input directory
files = list(INPUT_PATH.glob("dataset_*.pt"))
for filename in tqdm(files):
    dataloader = DataLoader(torch.load(filename), batch_size=1)  # get one at a time
    region_data = []
    skip_count = 0
    for batch in dataloader:
        # Try conversion. If it fails, skip this image
        try:
            region_cat = tile_cat_to_region_cat(TileCatalog(4, batch["tile_catalog"]), OVERLAP_SLEN)
        except:  # pylint:disable=bare-except  # noqa
            skip_count += 1
            continue

        batch["tile_catalog"] = region_cat.to_dict()
        region_data.append(batch)

    print("Skipped {skip_count} out of {len(dataloader)} total images")  # noqa: WPS421

    # Save converted data to new file
    output_filename = OUTPUT_PATH / filename.parts[-1]
    with open(output_filename, "wb") as f:
        torch.save(itemize_data(region_data), f)
