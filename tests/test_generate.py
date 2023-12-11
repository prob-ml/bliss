import os

import torch

from bliss.main import generate


class TestGenerate:
    def test_generate_sdss(self, cfg):
        # check that cached dataset exists
        assert cfg.generate.n_image_files > 0 and cfg.generate.n_batches_per_file > 0

        generate(cfg.generate)

        file_path = cfg.generate.cached_data_path + "/dataset_0.pt"
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
            correct_len = cfg.generate.n_batches_per_file * cfg.generate.simulator.prior.batch_size
            assert len(cached_dataset) == correct_len, (
                f"cached_dataset has length {len(cached_dataset)}, "
                f"but must be list of length {correct_len}"
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
