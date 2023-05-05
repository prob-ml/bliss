import os
import pickle
from typing import Dict, List, TypedDict

import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.utils.data import IterableDataset
from tqdm import tqdm

from bliss.catalog import TileCatalog

FileDatum = TypedDict(
    "FileDatum",
    {"tile_catalog": TileCatalog, "images": torch.Tensor, "background": torch.Tensor},
)


def generate(cfg: DictConfig):
    simulator = instantiate(cfg.simulator)

    file_data_capacity = cfg.cached_simulator.file_data_capacity
    cached_data_path = cfg.cached_simulator.cached_data_path

    # largest `batch_size` multiple <= `file_data_capacity`
    file_data_size = (
        file_data_capacity // simulator.image_prior.batch_size
    ) * simulator.image_prior.batch_size

    assert (
        file_data_size >= simulator.image_prior.batch_size
        and file_data_size > simulator.image_prior.batch_size * simulator.num_workers
    ), "file_data_capacity too small"

    # number of files needed to store >= `n_batches` * `batch_size` images
    # in <= `file_data_size`-image files
    n_files = -(
        simulator.n_batches * simulator.image_prior.batch_size // -file_data_size
    )  # ceil division

    # stores details of the written image files - { filename: string, data }
    data_files: List[Dict] = []

    # use SimulatedDataset to generate data in minibatches iteratively,
    # then concatenate before caching to disk via pickle
    simulated_dataset = simulator.train_dataloader().dataset
    assert isinstance(
        simulated_dataset, IterableDataset
    ), "simulated_dataset must be IterableDataset"

    # create cached_data_path if it doesn't exist
    if not os.path.exists(cached_data_path):
        os.makedirs(cached_data_path)

    # assume overwriting any existing cached image files
    for file_idx in tqdm(range(n_files), desc="Generating and writing cached dataset"):
        batch_data = []
        for _ in range(file_data_size // simulator.image_prior.batch_size):
            batch_data.append(next(iter(simulated_dataset)))

        # TODO: refactor/optimize this dictionary/tensor flattening
        flat_data = {}
        # concatenate tensors in tile_catalog dictionaries
        tile_catalog_flattened = {
            key: flatten_tile_catalog_tensor(key, batch_data)
            for key in batch_data[0]["tile_catalog"].keys()
        }
        flat_data["tile_catalog"] = tile_catalog_flattened
        flat_data["images"] = flatten_tensor("images", batch_data)
        flat_data["background"] = flatten_tensor("background", batch_data)

        assert len(flat_data["images"]) == file_data_size
        assert len(flat_data["background"]) == file_data_size
        # reconstruct data as list of single-input FileDatum dictionaries
        file_data: List[FileDatum] = []
        for i in range(file_data_size):
            file_datum: FileDatum = {}  # type: ignore
            # construct a TileCatalog dictionary of ith-input tensors
            file_datum["tile_catalog"] = {  # type: ignore
                k: flat_data["tile_catalog"][k][i] for k in flat_data["tile_catalog"].keys()
            }
            file_datum["images"] = flat_data["images"][i]
            file_datum["background"] = flat_data["background"][i]
            file_data.append(file_datum)

        data_files.append({"filename": f"dataset_{file_idx}.pkl", "data": file_data})
        with open(f"{cached_data_path}/{data_files[-1]['filename']}", "wb") as f:
            pickle.dump(file_data, f)


def flatten_tile_catalog_tensor(key, batch_data):
    if len(batch_data) > 1:
        flattened = torch.cat([data["tile_catalog"][key] for data in batch_data])
        flattened = torch.flatten(flattened, start_dim=0, end_dim=1)
    else:
        # only one batch, no need to cat / flatten
        flattened = batch_data[0]["tile_catalog"][key]
    return flattened


def flatten_tensor(key, batch_data):
    if len(batch_data) > 1:
        flattened = torch.cat([data[key] for data in batch_data])
        flattened = torch.flatten(flattened, start_dim=0, end_dim=1)
    else:
        # only one batch, no need to cat / flatten
        flattened = batch_data[0][key]
    return flattened
