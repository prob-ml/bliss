import collections
import copy
import logging
import pathlib
from typing import List

import numpy as np
import pandas as pd
import torch
from astropy.io import fits
from astropy.io.fits import Header
from astropy.wcs import WCS
from einops import rearrange
from pathos import multiprocessing

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
        min_flux_for_loss: int,
        prepare_data_processes_num: int,
        data_in_one_cached_file: int,
        splits: str,
        batch_size: int,
        num_workers: int,
        cached_data_path: str,
        train_transforms: List,
        nontrain_transforms: List,
        subset_fraction: float = None,
    ):
        super().__init__(
            splits,
            batch_size,
            num_workers,
            cached_data_path,
            train_transforms,
            nontrain_transforms,
            subset_fraction,
        )

        self.dc2_image_dir = dc2_image_dir
        self.dc2_cat_path = dc2_cat_path

        self.image_lim = image_lim
        self.n_image_split = n_image_split
        self.tile_slen = tile_slen
        self.max_sources_per_tile = max_sources_per_tile
        self.min_flux_for_loss = min_flux_for_loss
        self.prepare_data_processes_num = prepare_data_processes_num
        self.data_in_one_cached_file = data_in_one_cached_file

        assert (
            self.image_lim[0] % self.n_image_split == 0
        ), "image_lim is not divisible by n_image_split"
        assert (
            self.image_lim[1] % self.n_image_split == 0
        ), "image_lim is not divisible by n_image_split"
        assert (
            self.image_lim[0] == self.image_lim[1]
        ), "image_lim[0] should be equal to image_lim[1]"
        assert (self.image_lim[0] // self.n_image_split) % self.tile_slen == 0, "invalid tile_slen"
        assert (self.image_lim[1] // self.n_image_split) % self.tile_slen == 0, "invalid tile_slen"

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

        generate_cached_data_kwargs = {
            "image_files": self._image_files,
            "bg_files": self._bg_files,
            "image_lim": self.image_lim,
            "n_image_split": self.n_image_split,
            "tile_slen": self.tile_slen,
            "max_sources_per_tile": self.max_sources_per_tile,
            "bands": self.bands,
            "n_bands": self.n_bands,
            "dc2_cat_path": self.dc2_cat_path,
            "cached_data_path": self.cached_data_path,
            "min_flux_for_loss": self.min_flux_for_loss,
            "data_in_one_cached_file": self.data_in_one_cached_file,
        }
        generate_cached_data_wrapper = lambda image_index: generate_cached_data(
            image_index,
            **generate_cached_data_kwargs,
        )

        if self.prepare_data_processes_num > 1:
            with multiprocessing.Pool(processes=self.prepare_data_processes_num) as process_pool:
                process_pool.map(
                    generate_cached_data_wrapper,
                    list(range(n_image)),
                    chunksize=4,
                )
        else:
            for i in range(n_image):
                generate_cached_data_wrapper(i)

        return None

    def get_plotting_sample(self, image_index):
        if self._image_files is None or self._bg_files is None:
            self._load_image_and_bg_files_list()

        if image_index < 0 or image_index >= len(self._image_files[0]):
            raise IndexError("invalid image_idx")

        kwargs = {
            "image_files": self._image_files,
            "bg_files": self._bg_files,
            "image_lim": self.image_lim,
            "n_image_split": self.n_image_split,
            "tile_slen": self.tile_slen,
            "max_sources_per_tile": self.max_sources_per_tile,
            "bands": self.bands,
            "n_bands": self.n_bands,
            "dc2_cat_path": self.dc2_cat_path,
            "cached_data_path": self.cached_data_path,
            "min_flux_for_loss": self.min_flux_for_loss,
            "data_in_one_cached_file": self.data_in_one_cached_file,
        }

        result_dict = load_image_and_catalog(image_index, **kwargs)
        return {
            "tile_catalog": result_dict["tile_dict"],
            "image": result_dict["inputs"]["image"],
            "background": result_dict["inputs"]["bg"],
            "match_id": result_dict["other_info"]["match_id"],
            "full_catalog": result_dict["other_info"]["full_cat"],
            "wcs": result_dict["other_info"]["wcs"],
            "psf_params": result_dict["inputs"]["psf_params"],
        }


def squeeze_tile_dict(tile_dict):
    tile_dict_copy = copy.copy(tile_dict)
    for k, v in tile_dict_copy.items():
        if k != "n_sources":
            tile_dict_copy[k] = rearrange(v, "1 nth ntw s k -> nth ntw s k")
    tile_dict_copy["n_sources"] = rearrange(tile_dict_copy["n_sources"], "1 nth ntw -> nth ntw")
    return tile_dict_copy


def unsqueeze_tile_dict(tile_dict):
    tile_dict_copy = copy.copy(tile_dict)
    for k, v in tile_dict_copy.items():
        if k != "n_sources":
            tile_dict_copy[k] = rearrange(v, "nth ntw s k -> 1 nth ntw s k")
    tile_dict_copy["n_sources"] = rearrange(tile_dict_copy["n_sources"], "nth ntw -> 1 nth ntw")
    return tile_dict_copy


def load_image_and_catalog(image_index, **kwargs):
    image, bg, wcs_header_str = read_image_for_bands(image_index, **kwargs)
    wcs = wcs_from_wcs_header_str(wcs_header_str)

    plocs_lim = image[0].shape
    height = plocs_lim[0]
    width = plocs_lim[1]
    full_cat, psf_params, match_id = DC2FullCatalog.from_file(
        kwargs["dc2_cat_path"],
        wcs,
        height,
        width,
        bands=kwargs["bands"],
        n_bands=kwargs["n_bands"],
        min_flux_for_loss=kwargs["min_flux_for_loss"],
    )
    tile_cat = full_cat.to_tile_catalog(kwargs["tile_slen"], kwargs["max_sources_per_tile"])
    tile_dict = squeeze_tile_dict(tile_cat.data)

    # add one/two/more_than_two source mask
    on_mask = rearrange(tile_cat.is_on_mask, "1 nth ntw s -> nth ntw s 1")
    on_mask_count = on_mask.sum(dim=(-2, -1))
    tile_dict["one_source_mask"] = rearrange(on_mask_count == 1, "nth ntw -> nth ntw 1 1") & on_mask
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
            "bg": bg,
            "psf_params": psf_params,
        },
        "other_info": {
            "full_cat": full_cat,
            "wcs": wcs,
            "wcs_header_str": wcs_header_str,
            "match_id": match_id,
        },
    }


def generate_cached_data(image_index, **kwargs):
    result_dict = load_image_and_catalog(image_index, **kwargs)

    image = result_dict["inputs"]["image"]
    bg = result_dict["inputs"]["bg"]
    tile_dict = result_dict["tile_dict"]
    wcs_header_str = result_dict["other_info"]["wcs_header_str"]
    psf_params = result_dict["inputs"]["psf_params"]

    # split image
    split_lim = kwargs["image_lim"][0] // kwargs["n_image_split"]
    image_splits = split_tensor(image, split_lim, 1, 2)
    image_width_pixels = image.shape[2]
    split_image_num_on_width = image_width_pixels // split_lim
    bg_splits = split_tensor(bg, split_lim, 1, 2)

    # split tile cat
    tile_cat_splits = {}
    param_list = [
        "locs",
        "n_sources",
        "source_type",
        "galaxy_fluxes",
        "star_fluxes",
        "redshifts",
        "one_source_mask",
        "two_sources_mask",
        "more_than_two_sources_mask",
    ]
    for param_name in param_list:
        tile_cat_splits[param_name] = split_tensor(
            tile_dict[param_name], split_lim // kwargs["tile_slen"], 0, 1
        )

    data_splits = {
        "tile_catalog": unpack_dict(tile_cat_splits),
        "images": image_splits,
        "image_height_index": (
            torch.arange(0, len(image_splits)) // split_image_num_on_width
        ).tolist(),
        "image_width_index": (
            torch.arange(0, len(image_splits)) % split_image_num_on_width
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
        assert data_count < 1e5 and image_index < 1e5, "too many cached data files"
        assert len(tmp_data_cached) < 1e5, "too many cached data in one file"
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
        flux_r_band = catalog["flux_r"].values
        catalog = catalog.loc[flux_r_band > kwargs["min_flux_for_loss"]]

        objid = torch.from_numpy(catalog["id"].values)
        match_id = torch.from_numpy(catalog["match_objectId"].values)
        ra = torch.from_numpy(catalog["ra"].values).squeeze()
        dec = torch.from_numpy(catalog["dec"].values).squeeze()
        galaxy_bools = torch.from_numpy((catalog["truth_type"] == 1).values)
        star_bools = torch.from_numpy((catalog["truth_type"] == 2).values)
        flux, psf_params = get_bands_flux_and_psf(kwargs["bands"], catalog)
        do_have_redshifts = catalog.get("redshifts", "")
        initial_redshifts = torch.zeros_like(objid)
        redshifts = (
            torch.tensor(catalog["redshifts"].values) if do_have_redshifts else initial_redshifts
        )

        star_galaxy_filter = galaxy_bools | star_bools
        objid = objid[star_galaxy_filter]
        match_id = match_id[star_galaxy_filter]
        ra = ra[star_galaxy_filter]
        dec = dec[star_galaxy_filter]
        source_type = torch.from_numpy(catalog["truth_type"].values[star_galaxy_filter])
        source_type = torch.where(source_type == 2, SourceType.STAR, SourceType.GALAXY)
        star_fluxes = flux[star_galaxy_filter]
        galaxy_fluxes = flux[star_galaxy_filter]
        redshifts = redshifts[star_galaxy_filter] if do_have_redshifts else initial_redshifts

        plocs = cls.plocs_from_ra_dec(ra, dec, wcs).squeeze(0)
        x0_mask = (plocs[:, 0] > 0) & (plocs[:, 0] < height)
        x1_mask = (plocs[:, 1] > 0) & (plocs[:, 1] < width)
        plocs_mask = x0_mask * x1_mask

        objid = objid[plocs_mask]
        match_id = match_id[plocs_mask]
        plocs = plocs[plocs_mask]
        source_type = source_type[plocs_mask]
        star_fluxes = star_fluxes[plocs_mask]
        galaxy_fluxes = galaxy_fluxes[plocs_mask]
        redshifts = redshifts[plocs_mask] if do_have_redshifts else initial_redshifts

        nobj = source_type.shape[0]
        d = {
            "objid": objid.reshape(1, nobj, 1),
            "n_sources": torch.tensor((nobj,)),
            "source_type": source_type.reshape(1, nobj, 1),
            "plocs": plocs.reshape(1, nobj, 2),
            "redshifts": redshifts.reshape(1, nobj, 1),
            "galaxy_fluxes": galaxy_fluxes.reshape(1, nobj, kwargs["n_bands"]),
            "star_fluxes": star_fluxes.reshape(1, nobj, kwargs["n_bands"]),
        }

        return cls(height, width, d), psf_params, match_id


def read_image_for_bands(image_index, **kwargs):
    image_list = []
    bg_list = []
    wcs_header_str = None
    for b in range(kwargs["n_bands"]):
        image_frame = fits.open(kwargs["image_files"][b][image_index])
        bg_frame = fits.open(kwargs["bg_files"][b][image_index])
        image_data = image_frame[1].data
        bg_data = bg_frame[0].data

        if wcs_header_str is None:
            wcs_header_str = image_frame[1].header.tostring()

        image_frame.close()
        bg_frame.close()

        image = torch.nan_to_num(
            torch.from_numpy(image_data)[: kwargs["image_lim"][0], : kwargs["image_lim"][1]]
        )
        bg = torch.from_numpy(bg_data.astype(np.float32)).expand(
            kwargs["image_lim"][0], kwargs["image_lim"][1]
        )

        image += bg
        image_list.append(image)
        bg_list.append(bg)

    return torch.stack(image_list), torch.stack(bg_list), wcs_header_str


def get_bands_flux_and_psf(bands, catalog):
    flux_list = []
    psf_params_list = []
    for b in bands:
        flux_list.append(torch.from_numpy((catalog["flux_" + b]).values))
        psf_params_name = ["IxxPSF_pixel_", "IyyPSF_pixel_", "IxyPSF_pixel_", "psf_fwhm_"]
        psf_params_cur_band = []
        for i in psf_params_name:
            median_psf = np.nanmedian((catalog[i + b]).values).astype(np.float32)
            psf_params_cur_band.append(median_psf)
        psf_params_list.append(torch.tensor(psf_params_cur_band))

    return torch.stack(flux_list).t(), torch.stack(psf_params_list)
