# pylint: disable=R0801
import logging
import multiprocessing
import pathlib
import pickle
import re
from pathlib import Path

import torch

from bliss.surveys.dc2 import DC2DataModule, map_nested_dicts, split_list, unpack_dict


class RedshiftDC2DataModule(DC2DataModule):
    BANDS = ("u", "g", "r", "i", "z", "y")

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.dc2_image_dir = Path(self.dc2_image_dir)
        self.dc2_cat_path = Path(self.dc2_cat_path)
        self._tract_patches = None

    def setup(self, stage: str) -> None:  # noqa: WPS324
        """Setup following super(), but we save train/val/test splits to text files for ease."""
        super().setup(stage)

        # Log tracts and patches used
        train_files = self.file_paths[self.slices[0]]
        val_files = self.file_paths[self.slices[1]]
        test_files = self.file_paths[self.slices[2]]

        train_matches = [
            re.search(r"cached_data_(\d+)_(\d+,\d+)", file_path) for file_path in train_files
        ]
        val_matches = [
            re.search(r"cached_data_(\d+)_(\d+,\d+)", file_path) for file_path in val_files
        ]
        test_matches = [
            re.search(r"cached_data_(\d+)_(\d+,\d+)", file_path) for file_path in test_files
        ]
        train_tract_patches = [(match.group(1), match.group(2)) for match in train_matches]
        val_tract_patches = [(match.group(1), match.group(2)) for match in val_matches]
        test_tract_patches = [(match.group(1), match.group(2)) for match in test_matches]

        with open(self.cached_data_path / "train_splits.pkl", "wb") as f:
            pickle.dump(train_tract_patches, f)

        with open(self.cached_data_path / "val_splits.pkl", "wb") as f:
            pickle.dump(val_tract_patches, f)

        with open(self.cached_data_path / "test_splits.pkl", "wb") as f:
            pickle.dump(test_tract_patches, f)

    def _load_image_and_bg_files_list(self):
        img_pattern = "**/*/calexp*.fits"
        bg_pattern = "**/*/bkgd*.fits"
        image_files = []
        bg_files = []

        for band in self.bands:
            band_path = self.dc2_image_dir / str(band)
            img_file_list = list(pathlib.Path(band_path).glob(img_pattern))
            bg_file_list = list(pathlib.Path(band_path).glob(bg_pattern))

            image_files.append(sorted(img_file_list))
            bg_files.append(sorted(bg_file_list))
        n_image = len(bg_files[0])

        # assign state only in main process
        self._image_files = image_files
        self._bg_files = bg_files

        # record which tracts and patches
        tracts = [str(file_name).split("/")[-3] for file_name in self._image_files[0]]
        patches = [
            str(file_name).rsplit("-", maxsplit=1)[-1][:3] for file_name in self._image_files[0]
        ]  # TODO: check
        self._tract_patches = [x[0] + "_" + x[1] for x in zip(tracts, patches)]  # TODO: hack

        return n_image

    def prepare_data(self):  # noqa: WPS324
        if self.cached_data_path.exists():
            logger = logging.getLogger("DC2DataModule")
            warning_msg = "WARNING: cached data already exists at [%s], we directly use it\n"
            logger.warning(warning_msg, str(self.cached_data_path))
            return None

        logger = logging.getLogger("DC2DataModule")
        warning_msg = "WARNING: can't find cached data, we generate it at [%s]\n"
        logger.warning(warning_msg, str(self.cached_data_path))
        if not self.cached_data_path.exists():
            self.cached_data_path.mkdir(parents=True)

        n_image = self._load_image_and_bg_files_list()

        # Train
        if self.prepare_data_processes_num > 1:
            with multiprocessing.Pool(processes=self.prepare_data_processes_num) as process_pool:
                process_pool.map(
                    self.generate_cached_data,
                    zip(list(range(n_image)), self._tract_patches),
                    chunksize=4,
                )
        else:
            for i in range(n_image):
                self.generate_cached_data((i, self._tract_patches[i]))

        return None

    def generate_cached_data(self, naming_info: tuple):  # pylint: disable=W0237,R0801
        image_index, patch_name = naming_info
        result_dict = self.load_image_and_catalog(image_index)

        image = result_dict["inputs"]["image"]
        tile_dict = result_dict["tile_dict"]
        wcs_header_str = result_dict["other_info"]["wcs_header_str"]
        psf_params = result_dict["inputs"]["psf_params"]

        param_list = [
            "locs",
            "n_sources",
            "source_type",
            "fluxes",
            "redshifts",
            "blendedness",
            "shear",
            "ellipticity",
            "cosmodc2_mask",
            "one_source_mask",
            "two_sources_mask",
            "more_than_two_sources_mask",
        ]

        splits = self.split_image_and_tile_cat(image, tile_dict, param_list, psf_params)

        data_splits = split_list(
            unpack_dict(splits),
            sub_list_len=self.data_in_one_cached_file,
        )

        data_count = 0
        for sub_splits in data_splits:  # noqa: WPS426
            tmp_data_cached = []
            for split in sub_splits:  # noqa: WPS426
                split_clone = map_nested_dicts(
                    split, lambda x: x.clone() if isinstance(x, torch.Tensor) else x
                )
                split_clone.update(wcs_header_str=wcs_header_str)
                tmp_data_cached.append(split_clone)
            assert data_count < 1e5 and image_index < 1e5, "too many cached data files"
            assert len(tmp_data_cached) < 1e5, "too many cached data in one file"
            cached_data_file_name = (
                f"cached_data_{patch_name}_{data_count:04d}_size_{len(tmp_data_cached):04d}.pt"
            )
            cached_data_file_path = self.cached_data_path / cached_data_file_name
            with open(cached_data_file_path, "wb") as cached_data_file:
                torch.save(tmp_data_cached, cached_data_file)
            data_count += 1
