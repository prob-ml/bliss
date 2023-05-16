import os
from typing import Dict, List, TypedDict

import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import IterableDataset
from tqdm import tqdm

from bliss.catalog import TileCatalog

FileDatum = TypedDict(
    "FileDatum",
    {"tile_catalog": TileCatalog, "images": torch.Tensor, "background": torch.Tensor},
)


def generate(cfg: DictConfig):
    max_images_per_file = cfg.generate.max_images_per_file
    cached_data_path = cfg.generate.cached_data_path

    # largest `batch_size` multiple <= `max_images_per_file`
    bs = cfg.generate.batch_size
    images_per_file = (max_images_per_file // bs) * bs
    assert images_per_file >= bs, "max_images_per_file too small"

    # number of files needed to store >= `n_batches` * `batch_size` images
    # in <= `images_per_file`-image files
    n_files = -(cfg.generate.n_batches * bs // -images_per_file)  # ceil division

    # use SimulatedDataset to generate data in minibatches iteratively,
    # then concatenate before caching to disk via pickle
    simulator = instantiate(cfg.simulator, prior={"batch_size": bs})
    simulated_dataset = simulator.train_dataloader().dataset
    assert isinstance(
        simulated_dataset, IterableDataset
    ), "simulated_dataset must be IterableDataset"

    # create cached_data_path if it doesn't exist
    if not os.path.exists(cached_data_path):
        os.makedirs(cached_data_path)
    print("Data will be saved to {}".format(cached_data_path))

    # Save Hydra config (used to generate data) to cached_data_path
    with open(f"{cfg.generate.cached_data_path}/hparams.yaml", "w", encoding="utf-8") as f:
        OmegaConf.save(cfg, f)

    if "train" in cfg.generate.splits:
        # assume overwriting any existing cached image files
        for file_idx in tqdm(range(n_files), desc="Generating and writing cached dataset files"):
            batch_data = generate_data(
                images_per_file // bs, simulated_dataset, "Simulating images in batches for file"
            )
            file_data = itemize_data(batch_data)
            with open(f"{cached_data_path}/dataset_{file_idx}.pt", "wb") as f:
                torch.save(file_data, f)

    if "valid" in cfg.generate.splits:
        valid = generate_data(
            cfg.generate.valid_n_batches,
            simulated_dataset,
            "Generating fixed validation set in batches",
        )
        with open(f"{cached_data_path}/dataset_valid.pt", "wb") as f:
            torch.save(valid, f)

    if "test" in cfg.generate.splits:
        test = generate_data(
            cfg.generate.test_n_batches,
            simulated_dataset,
            "Generating fixed test set in batches",
        )
        with open(f"{cached_data_path}/dataset_test.pt", "wb") as f:
            torch.save(test, f)


def generate_data(n_batches: int, simulated_dataset, desc="Generating data"):
    batch_data: List[Dict[str, torch.Tensor]] = []
    for _ in tqdm(range(n_batches), desc=desc):
        batch_data.append(next(iter(simulated_dataset)))
    return batch_data


def itemize_data(batch_data) -> List[FileDatum]:
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

    # reconstruct data as list of single-input FileDatum dictionaries
    n_items = len(flat_data["images"])
    file_data: List[FileDatum] = []
    for i in range(n_items):
        file_datum: FileDatum = {}  # type: ignore
        # construct a TileCatalog dictionary of ith-input tensors
        file_datum["tile_catalog"] = {  # type: ignore
            k: flat_data["tile_catalog"][k][i] for k in flat_data["tile_catalog"].keys()
        }
        file_datum["images"] = flat_data["images"][i]
        file_datum["background"] = flat_data["background"][i]
        file_data.append(file_datum)

    return file_data


def flatten_tile_catalog_tensor(key, batch_data):
    if len(batch_data) > 1:
        flattened = torch.cat([data["tile_catalog"][key] for data in batch_data])
        flattened = torch.flatten(torch.unsqueeze(flattened, 0), start_dim=0, end_dim=1)
    else:
        # only one batch, no need to cat / flatten
        flattened = batch_data[0]["tile_catalog"][key]
    return flattened


def flatten_tensor(key, batch_data):
    if len(batch_data) > 1:
        flattened = torch.cat([data[key] for data in batch_data])
        flattened = torch.flatten(torch.unsqueeze(flattened, 0), start_dim=0, end_dim=1)
    else:
        # only one batch, no need to cat / flatten
        flattened = batch_data[0][key]
    return flattened
