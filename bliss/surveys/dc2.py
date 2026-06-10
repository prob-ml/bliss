import collections
import copy
import logging
import multiprocessing
import pathlib
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch
from astropy.io import fits
from astropy.io.fits import Header
from astropy.wcs import WCS
from einops import rearrange

from bliss.cached_dataset import CachedSimulatedDataModule
from bliss.catalog import FullCatalog, SourceType


def wcs_from_wcs_header_str(wcs_header_str: str):
    return WCS(Header.fromstring(wcs_header_str))


def map_nested_dicts(cur_dict, func):
    if isinstance(cur_dict, collections.abc.Mapping):
        return {k: map_nested_dicts(v, func) for k, v in cur_dict.items()}
    return func(cur_dict)


def split_list(ori_list, sub_list_len):
    return [ori_list[i : (i + sub_list_len)] for i in range(0, len(ori_list), sub_list_len)]


def unpack_dict(ori_dict):
    return [dict(zip(ori_dict, v, strict=True)) for v in zip(*ori_dict.values(), strict=True)]


def split_tensor(
    ori_tensor: torch.Tensor, split_size: int, split_first_dim: int, split_second_dim: int
):
    tensor_splits = torch.stack(ori_tensor.split(split_size, dim=split_first_dim))
    tensor_splits = torch.stack(tensor_splits.split(split_size, dim=split_second_dim + 1), dim=1)
    return [sub_tensor.squeeze(0) for sub_tensor in tensor_splits.flatten(0, 1).split(1, dim=0)]


class DC2DataModule(CachedSimulatedDataModule):
    BANDS = ("u", "g", "r", "i", "z", "y")

    def __init__(
        self,
        dc2_image_dir: str,
        dc2_cat_path: str,
        image_lim: List[int],
        n_image_split: int,
        tile_slen: int,
        max_sources_per_tile: int,
        catalog_min_r_flux: float,
        prepare_data_processes_num: int,
        data_in_one_cached_file: int,
        splits: str,
        batch_size: int,
        num_workers: int,
        cached_data_path: str,
        train_transforms: List,
        nontrain_transforms: List,
        subset_fraction: float = None,
        shuffle_file_order: bool = True,
    ):
        super().__init__(
            splits,
            batch_size,
            num_workers,
            cached_data_path,
            train_transforms,
            nontrain_transforms,
            subset_fraction,
            shuffle_file_order,
        )

        self.dc2_image_dir = dc2_image_dir
        self.dc2_cat_path = dc2_cat_path

        self.image_lim = image_lim
        self.n_image_split = n_image_split
        self.tile_slen = tile_slen
        self.max_sources_per_tile = max_sources_per_tile
        self.catalog_min_r_flux = catalog_min_r_flux
        self.prepare_data_processes_num = prepare_data_processes_num
        self.data_in_one_cached_file = data_in_one_cached_file

        assert self.image_lim[0] == self.image_lim[1], (
            "image_lim[0] should be equal to image_lim[1]"
        )
        assert self.image_lim[0] % self.n_image_split == 0, (
            "image_lim is not divisible by n_image_split"
        )
        assert (self.image_lim[0] // self.n_image_split) % self.tile_slen == 0, "invalid tile_slen"

        self.bands = self.BANDS
        self.n_bands = len(self.BANDS)

        self._image_files = None
        self._bg_files = None

    def _load_image_and_bg_files_list(self):
        img_pattern = "**/*/calexp*.fits"
        bg_pattern = "**/*/bkgd*.fits"
        image_files = []
        bg_files = []

        for band in self.bands:
            band_path = self.dc2_image_dir + str(band)
            img_file_list = list(pathlib.Path(band_path).glob(img_pattern))
            bg_file_list = list(pathlib.Path(band_path).glob(bg_pattern))

            image_files.append(sorted(img_file_list))
            bg_files.append(sorted(bg_file_list))
        n_image = len(bg_files[0])

        # assign state only in main process
        self._image_files = image_files
        self._bg_files = bg_files

        return n_image

    def prepare_data(self):
        if self.cached_data_path.exists():
            logger = logging.getLogger("DC2DataModule")
            warning_msg = "WARNING: cached data already exists at [%s], we directly use it\n"
            logger.warning(warning_msg, str(self.cached_data_path))
            return None

        logger = logging.getLogger("DC2DataModule")
        warning_msg = "WARNING: can't find cached data, we generate it at [%s]\n"
        logger.warning(warning_msg, str(self.cached_data_path))
        self.cached_data_path.mkdir(parents=True)

        n_image = self._load_image_and_bg_files_list()

        if self.prepare_data_processes_num > 1:
            with multiprocessing.Pool(processes=self.prepare_data_processes_num) as process_pool:
                process_pool.map(
                    self.generate_cached_data,
                    list(range(n_image)),
                    chunksize=4,
                )
        else:
            for i in range(n_image):
                self.generate_cached_data(i)

        return None

    def get_plotting_sample(self, image_index):
        if self._image_files is None or self._bg_files is None:
            self._load_image_and_bg_files_list()

        if image_index < 0 or image_index >= len(self._image_files[0]):
            raise IndexError("invalid image_idx")

        result_dict = self.load_image_and_catalog(image_index)
        return {
            "tile_catalog": result_dict["tile_dict"],
            "image": result_dict["inputs"]["image"],
            "match_id": result_dict["other_info"]["match_id"],
            "full_catalog": result_dict["other_info"]["full_cat"],
            "wcs": result_dict["other_info"]["wcs"],
            "psf_params": result_dict["inputs"]["psf_params"],
        }

    @classmethod
    def squeeze_tile_dict(cls, tile_dict):
        tile_dict_copy = copy.copy(tile_dict)
        return {k: v.squeeze(0) for k, v in tile_dict_copy.items()}

    @classmethod
    def unsqueeze_tile_dict(cls, tile_dict):
        tile_dict_copy = copy.copy(tile_dict)
        return {k: v.unsqueeze(0) for k, v in tile_dict_copy.items()}

    def load_image_and_catalog(self, image_index):
        image, wcs_header_str = self.read_image_for_bands(image_index)
        wcs = wcs_from_wcs_header_str(wcs_header_str)

        plocs_lim = image[0].shape
        height = plocs_lim[0]
        width = plocs_lim[1]
        full_cat, psf_params, match_id = DC2FullCatalog.from_file(
            self.dc2_cat_path,
            wcs,
            height,
            width,
            bands=self.bands,
            n_bands=self.n_bands,
            catalog_min_r_flux=self.catalog_min_r_flux,
        )
        tile_cat = full_cat.to_tile_catalog(self.tile_slen, self.max_sources_per_tile)
        tile_dict = self.squeeze_tile_dict(tile_cat.data)

        # add one/two/more_than_two source mask
        on_mask = rearrange(tile_cat.is_on_mask, "1 nth ntw s -> nth ntw s 1")
        on_mask_count = on_mask.sum(dim=(-2, -1))
        tile_dict["one_source_mask"] = (
            rearrange(on_mask_count == 1, "nth ntw -> nth ntw 1 1") & on_mask
        )
        tile_dict["two_sources_mask"] = (
            rearrange(on_mask_count == 2, "nth ntw -> nth ntw 1 1") & on_mask
        )
        tile_dict["more_than_two_sources_mask"] = (
            rearrange(on_mask_count > 2, "nth ntw -> nth ntw 1 1") & on_mask
        )

        return {
            "tile_dict": tile_dict,
            "inputs": {
                "image": image,
                "psf_params": psf_params,
            },
            "other_info": {
                "full_cat": full_cat,
                "wcs": wcs,
                "wcs_header_str": wcs_header_str,
                "match_id": match_id,
            },
        }

    def split_image_and_tile_cat(self, image, tile_cat, tile_cat_keys_to_split, psf_params):
        # split image
        split_lim = self.image_lim[0] // self.n_image_split
        image_splits = split_tensor(image, split_lim, 1, 2)
        image_width_pixels = image.shape[2]
        split_image_num_on_width = image_width_pixels // split_lim

        # split tile cat
        tile_cat_splits = {}
        for param_name in tile_cat_keys_to_split:
            tile_cat_splits[param_name] = split_tensor(
                tile_cat[param_name], split_lim // self.tile_slen, 0, 1
            )

        return {
            "tile_catalog": unpack_dict(tile_cat_splits),
            "images": image_splits,
            "image_height_index": (
                torch.arange(0, len(image_splits)) // split_image_num_on_width
            ).tolist(),
            "image_width_index": (
                torch.arange(0, len(image_splits)) % split_image_num_on_width
            ).tolist(),
            "psf_params": [psf_params for _ in range(self.n_image_split**2)],
        }

    def generate_cached_data(self, image_index):
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

        for data_count, sub_splits in enumerate(data_splits):
            tmp_data_cached = []
            for split in sub_splits:
                split_clone = map_nested_dicts(
                    split, lambda x: x.clone() if isinstance(x, torch.Tensor) else x
                )
                split_clone.update(wcs_header_str=wcs_header_str)
                tmp_data_cached.append(split_clone)
            assert data_count < 1e5 and image_index < 1e5, "too many cached data files"
            assert len(tmp_data_cached) < 1e5, "too many cached data in one file"
            cached_data_file_name = (
                f"cached_data_{image_index:04d}_{data_count:04d}_size_{len(tmp_data_cached):04d}.pt"
            )
            cached_data_file_path = self.cached_data_path / cached_data_file_name
            with open(cached_data_file_path, "wb") as cached_data_file:
                torch.save(tmp_data_cached, cached_data_file)

    def read_image_for_bands(self, image_index):
        image_list = []
        wcs_header_str = None
        for b in range(self.n_bands):
            image_frame = fits.open(self._image_files[b][image_index])
            image_data = image_frame[1].data
            if wcs_header_str is None:
                wcs_header_str = image_frame[1].header.tostring()
            image_frame.close()

            image = torch.nan_to_num(
                torch.from_numpy(image_data)[: self.image_lim[0], : self.image_lim[1]]
            )
            # we assume image doesn't contain bg
            image_list.append(image)

        return torch.stack(image_list), wcs_header_str


class DC2FullCatalog(FullCatalog):
    @classmethod
    def from_file(cls, cat_path, wcs, height, width, **kwargs):
        cat_path = Path(cat_path)
        suffix = cat_path.suffix.lower()

        if suffix == ".parquet":
            catalog = pd.read_parquet(cat_path)
        elif suffix in (".pkl", ".pickle"):
            catalog = pd.read_pickle(cat_path)  # noqa: S301
        else:
            raise ValueError(f"Unsupported catalog file format: {suffix}")

        catalog = catalog.loc[catalog["flux_r"].values > kwargs["catalog_min_r_flux"]]

        col_names = (
            "id",
            "match_objectId",
            "ra",
            "dec",
            "truth_type",
            "blendedness",
            "shear_1",
            "shear_2",
            "ellipticity_1_true",
            "ellipticity_2_true",
            "cosmodc2_mask",
            "redshifts",
        )
        cols = {name: torch.tensor(catalog[name].values) for name in col_names}

        plocs = cls.plocs_from_ra_dec(cols["ra"], cols["dec"], wcs).squeeze(0)
        truth_type = cols["truth_type"]
        star_galaxy_filter = (truth_type == 1) | (truth_type == 2)
        # we ignore the supernova
        source_type = torch.where(truth_type == 2, SourceType.STAR, SourceType.GALAXY)
        fluxes, psf_params = cls.get_bands_flux_and_psf(kwargs["bands"], catalog)
        shear = torch.stack((cols["shear_1"], cols["shear_2"]), dim=-1)
        ellipticity = torch.stack((cols["ellipticity_1_true"], cols["ellipticity_2_true"]), dim=-1)

        plocs_start_point = torch.tensor([0.0, 0.0])
        plocs_end_point = torch.tensor([height, width])
        in_bounds = ((plocs > plocs_start_point) & (plocs < plocs_end_point)).all(dim=-1)
        keep = star_galaxy_filter & in_bounds

        nobj = int(keep.sum())
        d = {
            "objid": cols["id"][keep].view(1, nobj, 1),
            "source_type": source_type[keep].view(1, nobj, 1),
            "plocs": plocs[keep].view(1, nobj, 2),
            "redshifts": cols["redshifts"][keep].view(1, nobj, 1),
            "fluxes": fluxes[keep].view(1, nobj, kwargs["n_bands"]),
            "blendedness": cols["blendedness"][keep].view(1, nobj, 1),
            "shear": shear[keep].view(1, nobj, 2),
            "ellipticity": ellipticity[keep].view(1, nobj, 2),
            "cosmodc2_mask": cols["cosmodc2_mask"][keep].view(1, nobj, 1),
        }
        match_id = cols["match_objectId"][keep]

        cosmodc2_mask = rearrange(d["cosmodc2_mask"], "1 nobj 1 -> nobj")
        shear = d["shear"]
        ellipticity = d["ellipticity"]
        assert (
            not torch.isnan(shear[:, cosmodc2_mask, :]).any()
            and not torch.isnan(ellipticity[:, cosmodc2_mask, :]).any()
        )
        assert (
            torch.isnan(shear[:, ~cosmodc2_mask, :]).all()
            and torch.isnan(ellipticity[:, ~cosmodc2_mask, :]).all()
        )

        d["n_sources"] = torch.tensor((nobj,))

        return cls(height, width, d), psf_params, match_id

    @classmethod
    def get_bands_flux_and_psf(cls, bands, catalog):
        psf_prefixes = ("IxxPSF_pixel_", "IyyPSF_pixel_", "IxyPSF_pixel_", "psf_fwhm_")
        fluxes = torch.tensor(catalog[[f"flux_{b}" for b in bands]].values)
        psf_cols = [f"{p}{b}" for b in bands for p in psf_prefixes]
        psf_vals = catalog[psf_cols].values.T.reshape(len(bands), len(psf_prefixes), -1)
        median_psf = np.nanmedian(psf_vals, axis=-1).astype(np.float32)
        psf_params = torch.tensor(median_psf).unsqueeze(0)
        return fluxes, psf_params
