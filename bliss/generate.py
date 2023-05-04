import os
import pickle
from typing import Dict, List, TypedDict

import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.utils.data import IterableDataset
from tqdm import tqdm

from bliss.catalog import TileCatalog


def generate(cfg: DictConfig):
    simulator = instantiate(cfg.simulator)

    file_data_capacity = cfg.cached_simulator.file_data_capacity
    cached_data_path = cfg.cached_simulator.cached_data_path

    # largest `batch_size` multiple <= `file_data_capacity`
    file_data_size = (
        file_data_capacity // simulator.image_prior.batch_size
    ) * simulator.image_prior.batch_size

    assert file_data_size >= simulator.image_prior.batch_size, "file_data_capacity too small"

    # number of files needed to store >= `n_batches` * `batch_size` images
    # in <= `file_data_size`-image files
    n_files = -(
        simulator.n_batches * simulator.image_prior.batch_size // -file_data_size
    )  # ceil division

    # stores details of the written image files - { filename: string, data }
    data_files: List[Dict] = []

    # assume overwriting existing files (if exist)
    # use SimulatedDataset to generate data in minibatches iteratively,
    # then concatenate before caching to disk via pickle
    simulated_dataset = simulator.train_dataloader().dataset
    assert isinstance(
        simulated_dataset, IterableDataset
    ), "simulated_dataset must be IterableDataset"

    # create cached_data_path if it doesn't exist
    if not os.path.exists(cached_data_path):
        os.makedirs(cached_data_path)

    if len(data_files) != n_files:
        # if no written image files, render (and write to disk) images
        for file_idx in tqdm(range(n_files), desc="Generating and writing cached dataset"):
            batch_data = []
            for _ in range(file_data_size // simulator.image_prior.batch_size):
                batch_data.append(next(iter(simulated_dataset)))

            # TODO: refactor/optimize this dictionary/tensor flattening
            FileDatum = TypedDict(  # noqa: N806
                "FileDatum",
                {"tile_catalog": TileCatalog, "images": torch.Tensor, "background": torch.Tensor},
            )
            file_data: List[FileDatum] = []
            flat_data = {}
            # concatenate tensors in tile_catalog dictionaries
            tile_catalog_flattened = {}
            for key in batch_data[0]["tile_catalog"].keys():
                tile_catalog_flattened[key] = (
                    torch.flatten(
                        torch.cat([data["tile_catalog"][key] for data in batch_data]),
                        start_dim=0,
                        end_dim=1,
                    )
                    if len(batch_data) > 1
                    else batch_data[0]["tile_catalog"][key]
                )
                flat_data["tile_catalog"] = tile_catalog_flattened
            for key in batch_data[0].keys():
                if key != "tile_catalog":
                    flat_data[key] = (
                        torch.flatten(
                            torch.cat([data[key] for data in batch_data]), start_dim=0, end_dim=1
                        )
                        if len(batch_data) > 1
                        else batch_data[0][key]
                    )
            assert len(flat_data["images"]) == file_data_size
            assert len(flat_data["background"]) == file_data_size
            for i in range(file_data_size):
                file_datum: FileDatum = {}  # type: ignore
                for key, _ in flat_data.items():
                    file_datum[key] = (  # type: ignore
                        {k: flat_data[key][k][i] for k in flat_data[key].keys()}
                        if key == "tile_catalog"
                        else flat_data[key][i]
                    )
                file_data.append(file_datum)

            data_files.append({"filename": f"dataset_{file_idx}.pkl", "data": file_data})
            with open(f"{cached_data_path}/{data_files[-1]['filename']}", "wb") as f:
                pickle.dump(file_data, f)
