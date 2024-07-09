import collections
import copy
import logging
import math

# from pathos import multiprocessing
import os
import pathlib
import sys
from typing import List

import numpy as np
import pandas as pd
import torch
from astropy.io import fits
from astropy.io.fits import Header
from astropy.wcs import WCS
from einops import rearrange

from bliss.cached_dataset import CachedSimulatedDataModule
from bliss.catalog import FullCatalog
from bliss.surveys.dc2 import (
    DC2DataModule,
    DC2FullCatalog,
    get_bands_flux_and_psf,
    map_nested_dicts,
    read_image_for_bands,
    split_list,
    split_tensor,
    squeeze_tile_dict,
    unpack_dict,
    unsqueeze_tile_dict,
    wcs_from_wcs_header_str,
)


class LensingDC2DataModule(DC2DataModule):
    def __init__(
        self,
        dc2_image_dir: str,
        dc2_cat_path: str,
        image_slen: int,  # assume square images: image_slen x image_slen
        n_image_split: int,
        tile_slen: int,
        splits: str,
        batch_size: int,
        num_workers: int,
        cached_data_path: str,
        **kwargs,
    ):
        super().__init__(
            dc2_image_dir=dc2_image_dir,
            dc2_cat_path=dc2_cat_path,
            image_lim=[image_slen, image_slen],
            n_image_split=n_image_split,
            tile_slen=tile_slen,
            max_sources_per_tile=tile_slen
            ** 2,  # max of one source per pixel, TODO: calc max sources per tile
            min_flux_for_loss=-sys.maxsize - 1,  # smaller than any int
            prepare_data_processes_num=1,
            data_in_one_cached_file=100000,
            splits=splits,
            batch_size=batch_size,
            num_workers=num_workers,
            cached_data_path=cached_data_path,
            train_transforms=[],
            nontrain_transforms=[],
            subset_fraction=None,
        )

        self.dc2_image_dir = dc2_image_dir
        self.dc2_cat_path = dc2_cat_path
        self.image_slen = image_slen
        self.n_image_split = n_image_split
        self.tile_slen = tile_slen
        self.bands = self.BANDS
        self.n_bands = len(self.BANDS)

    # _load_image_and_bg_files_list can stay the same

    # override prepare_data
    def prepare_data(self):
        if self.cached_data_path.exists():
            logger = logging.getLogger("LensingDC2DataModule")
            warning_msg = "WARNING: cached data already exists at [%s], we directly use it\n"
            logger.warning(warning_msg, str(self.cached_data_path))
            return None

        logger = logging.getLogger("LensingDC2DataModule")
        warning_msg = "WARNING: can't find cached data, we generate it at [%s]\n"
        logger.warning(warning_msg, str(self.cached_data_path))
        self.cached_data_path.mkdir(parents=True)

        n_image = self._load_image_and_bg_files_list()

        generate_data_input = {
            "image_files": self._image_files,
            "bg_files": self._bg_files,
            "image_lim": [self.image_slen, self.image_slen],
            "n_image_split": self.n_image_split,
            "tile_slen": self.tile_slen,
            "bands": self.bands,
            "n_bands": self.n_bands,
            "dc2_cat_path": self.dc2_cat_path,
            "cached_data_path": self.cached_data_path,
            "data_in_one_cached_file": 100000,
        }

        generate_data_wrapper = lambda image_index: generate_cached_data(
            image_index,
            **generate_data_input,
        )

        for i in range(n_image):
            generate_data_wrapper(i)


def load_image_and_catalog(image_index, **kwargs):
    image, bg, wcs_header_str = read_image_for_bands(image_index, **kwargs)
    wcs = wcs_from_wcs_header_str(wcs_header_str)

    plocs_lim = image[0].shape
    height = plocs_lim[0]
    width = plocs_lim[1]
    full_cat, psf_params = DC2FullCatalog.from_file(
        kwargs["dc2_cat_path"],
        wcs,
        height,
        width,
        bands=kwargs["bands"],
        n_bands=kwargs["n_bands"],
    )
    tile_cat = full_cat.to_tile_catalog(
        kwargs["tile_slen"], kwargs["tile_slen"] ** 2
    )  # max sources = all sources

    tile_dict = squeeze_tile_dict(tile_cat.data)

    return {
        "tile_dict": tile_dict,
        "inputs": {
            "image": image,
            "bg": bg,
            "psf_params": psf_params,
        },
        "other_info": {
            "full_cat": full_cat,
            "wcs": wcs,
            "wcs_header_str": wcs_header_str,
        },
    }


def generate_cached_data(image_index, **kwargs):
    result_dict = load_image_and_catalog(image_index, **kwargs)

    image = result_dict["inputs"]["image"]
    bg = result_dict["inputs"]["bg"]
    tile_dict = result_dict["tile_dict"]
    wcs_header_str = result_dict["other_info"]["wcs_header_str"]
    psf_params = result_dict["inputs"]["psf_params"]

    # set shear and convergence to the nonzero tile mean here
    # TODO: how to do interpolation? (not sure if necessary)
    shear = tile_dict["shear"]
    convergence = tile_dict["convergence"]
    nonzero_sh_conv_mask = convergence != 0

    avg_nonzero_convergence = torch.mean(convergence * nonzero_sh_conv_mask, axis=2)
    avg_nonzero_shear = torch.mean(shear * nonzero_sh_conv_mask, axis=2)

    tile_dict["shear"] = avg_nonzero_shear
    tile_dict["convergence"] = avg_nonzero_convergence

    # split image
    split_lim = kwargs["image_lim"][0] // kwargs["n_image_split"]
    image_splits = split_tensor(image, split_lim, 1, 2)
    image_height_pixels = image.shape[1]
    split_image_num_on_height = image_height_pixels // split_lim
    bg_splits = split_tensor(bg, split_lim, 1, 2)

    # split tile cat
    tile_cat_splits = {}
    param_list = ["locs", "n_sources", "shear", "convergence"]

    for param_name in param_list:
        tile_cat_splits[param_name] = split_tensor(
            tile_dict[param_name], split_lim // kwargs["tile_slen"], 0, 1
        )

    data_splits = {
        "tile_catalog": unpack_dict(tile_cat_splits),
        "images": image_splits,
        "image_height_index": (
            torch.arange(0, len(image_splits)) % split_image_num_on_height
        ).tolist(),
        "image_width_index": (
            torch.arange(0, len(image_splits)) // split_image_num_on_height
        ).tolist(),
        "background": bg_splits,
        "psf_params": [psf_params for _ in range(kwargs["n_image_split"] ** 2)],
    }
    data_splits = split_list(
        unpack_dict(data_splits),
        sub_list_len=kwargs["data_in_one_cached_file"],
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
        # assert data_count < 1e5 and image_index < 1e5, "too many cached data files"
        # assert len(tmp_data_cached) < 1e5, "too many cached data in one file"
        cached_data_file_name = (
            f"cached_data_{image_index:04d}_{data_count:04d}_size_{len(tmp_data_cached):04d}.pt"
        )
        with open(
            kwargs["cached_data_path"] / cached_data_file_name,
            "wb",
        ) as cached_data_file:
            torch.save(tmp_data_cached, cached_data_file)
        data_count += 1


class DC2FullCatalog(FullCatalog):
    @classmethod
    def from_file(cls, cat_path, wcs, height, width, **kwargs):
        catalog = pd.read_pickle(cat_path)

        objid = torch.from_numpy(catalog["id"].values)
        ra = torch.from_numpy(catalog["ra"].values).squeeze()
        dec = torch.from_numpy(catalog["dec"].values).squeeze()
        # galaxy_bools = torch.from_numpy((catalog["truth_type"] == 1).values)

        shear_1 = torch.from_numpy(catalog["shear_1"].values).squeeze()
        shear_2 = torch.from_numpy(catalog["shear_2"].values).squeeze()
        convergence = torch.from_numpy(catalog["convergence"].values)

        flux, psf_params = get_bands_flux_and_psf(kwargs["bands"], catalog)

        plocs = cls.plocs_from_ra_dec(ra, dec, wcs).squeeze(0)
        x0_mask = (plocs[:, 0] > 0) & (plocs[:, 0] < height)
        x1_mask = (plocs[:, 1] > 0) & (plocs[:, 1] < width)
        plocs_mask = x0_mask * x1_mask

        objid = objid[plocs_mask]
        plocs = plocs[plocs_mask]
        # galaxy_fluxes = galaxy_fluxes[plocs_mask]
        shear_1 = shear_1[plocs_mask]
        shear_2 = shear_2[plocs_mask]
        convergence = convergence[plocs_mask]
        shear = torch.hstack((shear_1, shear_2))

        nobj = objid.shape[0]
        d = {
            "objid": objid.reshape(1, nobj, 1),
            "n_sources": torch.tensor((nobj,)),
            "plocs": plocs.reshape(1, nobj, 2),
            # "galaxy_fluxes": galaxy_fluxes.reshape(1, nobj, kwargs["n_bands"]),
            "shear": shear.reshape(1, nobj, 2),
            "convergence": convergence.reshape(1, nobj, 1),
        }

        return cls(height, width, d), psf_params
