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
    return [dict(zip(ori_dict, v)) for v in zip(*ori_dict.values())]


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

        assert (
            self.image_lim[0] == self.image_lim[1]
        ), "image_lim[0] should be equal to image_lim[1]"
        assert (
            self.image_lim[0] % self.n_image_split == 0
        ), "image_lim is not divisible by n_image_split"
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

    def prepare_data(self):  # noqa: WPS324
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
                f"cached_data_{image_index:04d}_{data_count:04d}_size_{len(tmp_data_cached):04d}.pt"
            )
            cached_data_file_path = self.cached_data_path / cached_data_file_name
            with open(cached_data_file_path, "wb") as cached_data_file:
                torch.save(tmp_data_cached, cached_data_file)
            data_count += 1

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
        # load catalog from either a string path or a Path

        cat_path = Path(cat_path)
        suffix = cat_path.suffix.lower()

        if suffix == ".parquet":
            catalog = pd.read_parquet(cat_path)
        elif suffix in (".pkl", ".pickle"):
            catalog = pd.read_pickle(cat_path)
        else:
            raise ValueError(f"Unsupported catalog file format: {suffix}")

        flux_r_band = catalog["flux_r"].values
        catalog = catalog.loc[flux_r_band > kwargs["catalog_min_r_flux"]]

        objid = torch.from_numpy(catalog["id"].values)
        match_id = torch.from_numpy(catalog["match_objectId"].values)
        ra = torch.from_numpy(catalog["ra"].values)
        dec = torch.from_numpy(catalog["dec"].values)
        plocs = cls.plocs_from_ra_dec(ra, dec, wcs).squeeze(0)
        galaxy_bools = torch.from_numpy((catalog["truth_type"] == 1).values)
        star_bools = torch.from_numpy((catalog["truth_type"] == 2).values)
        source_type = torch.from_numpy(catalog["truth_type"].values)
        # we ignore the supernova
        source_type = torch.where(source_type == 2, SourceType.STAR, SourceType.GALAXY)
        fluxes, psf_params = cls.get_bands_flux_and_psf(kwargs["bands"], catalog)
        blendedness = torch.from_numpy(catalog["blendedness"].values)
        shear1 = torch.from_numpy(catalog["shear_1"].values)
        shear2 = torch.from_numpy(catalog["shear_2"].values)
        shear = torch.stack((shear1, shear2), dim=-1)
        ellipticity1 = torch.from_numpy(catalog["ellipticity_1_true"].values)
        ellipticity2 = torch.from_numpy(catalog["ellipticity_2_true"].values)
        ellipticity = torch.stack((ellipticity1, ellipticity2), dim=-1)
        cosmodc2_mask = torch.from_numpy(catalog["cosmodc2_mask"].values)
        redshifts = torch.from_numpy(catalog["redshifts"].values)

        ori_len = len(catalog)
        d = {
            "objid": objid.view(1, ori_len, 1),
            "source_type": source_type.view(1, ori_len, 1),
            "plocs": plocs.view(1, ori_len, 2),
            "redshifts": redshifts.view(1, ori_len, 1),
            "fluxes": fluxes.view(1, ori_len, kwargs["n_bands"]),
            "blendedness": blendedness.view(1, ori_len, 1),
            "shear": shear.view(1, ori_len, 2),
            "ellipticity": ellipticity.view(1, ori_len, 2),
            "cosmodc2_mask": cosmodc2_mask.view(1, ori_len, 1),
        }

        star_galaxy_filter = galaxy_bools | star_bools
        for k, v in d.items():
            d[k] = v[:, star_galaxy_filter, :]
        match_id = match_id[star_galaxy_filter]

        plocs_start_point = torch.tensor([0.0, 0.0]).view(1, 1, -1)
        plocs_end_point = torch.tensor([height, width]).view(1, 1, -1)
        plocs_mask = ((d["plocs"] > plocs_start_point) & (d["plocs"] < plocs_end_point)).all(dim=-1)
        plocs_mask = plocs_mask.squeeze(0)
        for k, v in d.items():
            d[k] = v[:, plocs_mask, :]
        match_id = match_id[plocs_mask]

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

        nobj = d["source_type"].shape[1]
        d["n_sources"] = torch.tensor((nobj,))

        return cls(height, width, d), psf_params, match_id

    @classmethod
    def get_bands_flux_and_psf(cls, bands, catalog, median=True):
        flux_list = []
        psf_params_list = []
        for b in bands:
            flux_list.append(torch.from_numpy((catalog[f"flux_{b}"]).values))
            psf_params_name = ["IxxPSF_pixel_", "IyyPSF_pixel_", "IxyPSF_pixel_", "psf_fwhm_"]
            psf_params_cur_band = []
            for i in psf_params_name:
                if median:
                    median_psf = np.nanmedian((catalog[f"{i}{b}"]).values).astype(np.float32)
                    psf_params_cur_band.append(median_psf)
                else:
                    psf_params_cur_band.append(catalog[f"{i}{b}"].values.astype(np.float32))
            psf_params_list.append(
                torch.tensor(psf_params_cur_band)
            )  # bands x 4 (params per band) x n_obj

        return torch.stack(flux_list).t(), torch.stack(psf_params_list).unsqueeze(0)
