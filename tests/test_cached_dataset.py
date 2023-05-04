import os
import pickle

import torch

from bliss.generate import generate
from bliss.train import train


class TestCachedDataset:
    def test_generate(self, cfg):
        generate(cfg)
        # check that cached dataset exists
        cached_dataset_should_exist = cfg.simulator.n_batches > 0 and (
            cfg.simulator.prior.batch_size < cfg.cached_simulator.file_data_capacity
        )
        file_path = cfg.cached_simulator.cached_data_path + "/dataset_0.pkl"
        if cached_dataset_should_exist:
            assert os.path.exists(file_path), f"{file_path} not found"
        # cursory check of contents of cached dataset
        with open(file_path, "rb") as f:
            cached_dataset = pickle.load(f)
            assert isinstance(cached_dataset, list), "cached_dataset must be a list"
            assert isinstance(
                cached_dataset[0], dict
            ), "cached_dataset must be a list of dictionaries"
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

    def test_train_with_cached_data(self, cfg):
        cfg.training.use_cached_simulator = True
        # TODO: if possible, check that training uses cached dataset
        train(cfg)
