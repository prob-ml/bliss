import os

import pytest
import torch

from bliss.generate import generate
from bliss.train import train


@pytest.fixture()
def cached_data(cfg):
    generate(cfg)
    # check that cached dataset exists
    cached_dataset_should_exist = cfg.simulator.n_batches > 0 and (
        cfg.simulator.prior.batch_size < cfg.generate.max_images_per_file
    )
    file_path = cfg.cached_simulator.cached_data_path + "/dataset_0.pt"
    if cached_dataset_should_exist:
        assert os.path.exists(file_path), f"{file_path} not found"
    # cursory check of contents of cached dataset
    with open(file_path, "rb") as f:
        cached_dataset = torch.load(f)
        assert isinstance(cached_dataset, list), "cached_dataset must be a list"
        assert isinstance(cached_dataset[0], dict), "cached_dataset must be a list of dictionaries"
        assert isinstance(
            cached_dataset[0]["tile_catalog"], dict
        ), "cached_dataset[0]['tile_catalog'] must be a dictionary"
        assert isinstance(
            cached_dataset[0]["images"], torch.Tensor
        ), "cached_dataset[0]['images'] must be a torch.Tensor"
        assert isinstance(
            cached_dataset[0]["background"], torch.Tensor
        ), "cached_dataset[0]['background'] must be a torch.Tensor"
        assert (
            len(cached_dataset) == cfg.simulator.prior.batch_size
        ), f"cached_dataset must be a list of length {cfg.simulator.prior.batch_size}"
        assert (
            len(cached_dataset[0]["images"]) == 1
        ), "cached_dataset[0]['images'] must be a single tensor"
        assert cached_dataset[0]["images"][0].shape == (
            cfg.simulator.prior.n_tiles_h * cfg.simulator.prior.tile_slen,
            cfg.simulator.prior.n_tiles_w * cfg.simulator.prior.tile_slen,
        )
        assert cached_dataset[0]["background"][0].shape == (
            cfg.simulator.prior.n_tiles_h * cfg.simulator.prior.tile_slen,
            cfg.simulator.prior.n_tiles_w * cfg.simulator.prior.tile_slen,
        )


class TestCachedDataset:
    # End-to-end test, using cached dataset.

    def test_train_with_cached_data(self, cfg, cached_data):
        cfg.training.use_cached_simulator = True
        # TODO: check that training uses cached dataset
        train(cfg)