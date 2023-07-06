import os
from typing import Dict, List, TypedDict

import torch
from einops import rearrange
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from bliss.catalog import TileCatalog

FileDatum = TypedDict(
    "FileDatum",
    {
        "tile_catalog": TileCatalog,
        "images": torch.Tensor,
        "background": torch.Tensor,
        "deconvolution": torch.Tensor,
        "psf_params": torch.Tensor,
    },
)


def generate(cfg: DictConfig):
    max_images_per_file = cfg.generate.max_images_per_file
    cached_data_path = cfg.generate.cached_data_path
    n_workers_per_process = cfg.generate.n_workers_per_process

    # largest `batch_size` multiple <= `max_images_per_file`
    bs = cfg.generate.batch_size
    images_per_file = (max_images_per_file // bs) * bs
    assert images_per_file >= bs, "max_images_per_file too small"

    # number of files needed to store >= `n_batches` * `batch_size` images
    # in <= `images_per_file`-image files
    n_files = -(cfg.generate.n_batches * bs // -images_per_file)  # ceil division

    # note: this is technically "n_files for this process"
    process_index = cfg.generate.get("process_index", 0)
    files_start_idx = process_index * n_files

    # use SimulatedDataset to generate data in minibatches iteratively,
    # then concatenate before caching to disk via pickle
    simulator = instantiate(
        cfg.simulator,
        num_workers=n_workers_per_process,
        survey={"prior_config": {"batch_size": bs}},
    )
    simulated_dataset = simulator.train_dataloader().dataset

    # create cached_data_path if it doesn't exist
    if not os.path.exists(cached_data_path):
        os.makedirs(cached_data_path)
    print("Data will be saved to {}".format(cached_data_path))  # noqa: WPS421

    # Save Hydra config (used to generate data) to cached_data_path
    with open(f"{cfg.generate.cached_data_path}/hparams.yaml", "w", encoding="utf-8") as f:
        OmegaConf.save(cfg, f)

    # assume overwriting any existing cached image files
    file_idxs = range(files_start_idx, files_start_idx + n_files)
    for file_idx in tqdm(file_idxs, desc="Generating and writing cached dataset files"):
        batch_data = generate_data(
            images_per_file // bs, simulated_dataset, "Simulating images in batches for file"
        )
        file_data = itemize_data(batch_data)
        with open(f"{cached_data_path}/{cfg.generate.file_prefix}_{file_idx}.pt", "wb") as f:
            torch.save(file_data, f)


def generate_data(n_batches: int, simulated_dataset, desc="Generating data"):
    batch_data: List[Dict[str, torch.Tensor]] = []
    for _ in tqdm(range(n_batches), desc=desc):
        batch_data.append(next(iter(simulated_dataset)))
    return batch_data


def itemize_data(batch_data) -> List[FileDatum]:
    flat_data = {}

    # flatten tile catalog
    tile_catalog_flattened = {}
    for key in batch_data[0]["tile_catalog"].keys():
        batch_tc_key = torch.stack([data["tile_catalog"][key] for data in batch_data])
        tile_catalog_flattened[key] = rearrange(batch_tc_key, "b c ... -> (b c) ...")
    flat_data["tile_catalog"] = tile_catalog_flattened

    # flatten the rest of the data
    keys = ["images", "background", "deconvolution", "psf_params"]
    for key in keys:
        if key in batch_data[0]:
            batch_ch = torch.stack([data[key] for data in batch_data])
            flat_data[key] = rearrange(batch_ch, "b c ... -> (b c) ...")

    # reconstruct data as list of single-input FileDatum dictionaries
    n_items = len(flat_data["images"])
    file_data: List[FileDatum] = []
    for i in range(n_items):
        file_datum: FileDatum = {}
        # construct a TileCatalog dictionary of ith-input tensors
        file_datum["tile_catalog"] = {
            k: flat_data["tile_catalog"][k][i] for k in flat_data["tile_catalog"].keys()
        }
        file_datum["images"] = flat_data["images"][i]
        file_datum["background"] = flat_data["background"][i]
        file_datum["deconvolution"] = flat_data["deconvolution"][i]
        file_datum["psf_params"] = flat_data["psf_params"][i]
        file_data.append(file_datum)

    return file_data
