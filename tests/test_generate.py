import os

import torch

from bliss.generate import generate


class TestGenerate:
    def test_generate(self, cfg):
        generate(cfg)
        # check that cached dataset exists
        cached_dataset_should_exist = cfg.generate.n_batches > 0 and (
            cfg.generate.batch_size < cfg.generate.max_images_per_file
        )
        file_path = cfg.generate.cached_data_path + "/dataset_0.pt"
        if cached_dataset_should_exist:
            assert os.path.exists(file_path), f"{file_path} not found"
        # cursory check of contents of cached dataset
        with open(file_path, "rb") as f:
            cached_dataset = torch.load(f)
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
            assert len(cached_dataset) == cfg.generate.max_images_per_file, (
                f"cached_dataset has length {len(cached_dataset)}, "
                f"but must be list of length {cfg.generate.max_images_per_file}"
            )
            assert (
                len(cached_dataset[0]["images"]) == 5
            ), "cached_dataset[0]['images'] must be a 5-D tensor"
            assert cached_dataset[0]["images"][0].shape == (
                cfg.simulator.prior.n_tiles_h * cfg.simulator.prior.tile_slen,
                cfg.simulator.prior.n_tiles_w * cfg.simulator.prior.tile_slen,
            )
            assert cached_dataset[0]["background"][0].shape == (
                cfg.simulator.prior.n_tiles_h * cfg.simulator.prior.tile_slen,
                cfg.simulator.prior.n_tiles_w * cfg.simulator.prior.tile_slen,
            )
