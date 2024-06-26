import collections
import copy
import logging
import math
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
from bliss.catalog import FullCatalog


def from_wcs_header_str_to_wcs(wcs_header_str: str):
    return WCS(Header.fromstring(wcs_header_str))


def map_nested_dicts(cur_dict, func):
    if isinstance(cur_dict, collections.abc.Mapping):
        return {k: map_nested_dicts(v, func) for k, v in cur_dict.items()}

    return func(cur_dict)


def split_list(ori_list, sub_list_len):
    return [ori_list[i : (i + sub_list_len)] for i in range(0, len(ori_list), sub_list_len)]


class DC2DataModule(CachedSimulatedDataModule):
    # why are these bands out of order? why does a test break if they are ordered correctly?
    BANDS = ("g", "i", "r", "u", "y", "z")

    def __init__(
        self,
        dc2_image_dir: str,
        dc2_cat_path: str,
        image_lim: List[int],
        n_image_split: int,
        min_flux_for_loss: int,
        prepare_data_processes_num: int,
        data_in_one_cached_file: int,
        splits: str,
        batch_size: int,
        num_workers: int,
        split_file_dir: str,
        split_seed: int = 0,
    ):
        super().__init__(
            data_dir=split_file_dir,
            splits=splits,
            split_seed=split_seed,
            batch_size=batch_size,
            num_workers=num_workers,
            convert_full_cat=False,
            tile_slen=4,
            max_sources=1,
        )

        self.dc2_image_dir = dc2_image_dir
        self.dc2_cat_path = dc2_cat_path

        self.image_lim = image_lim
        self.n_image_split = n_image_split
        self.min_flux_for_loss = min_flux_for_loss
        self.prepare_data_processes_num = prepare_data_processes_num
        self.data_in_one_cached_file = data_in_one_cached_file

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
        if self.data_dir.exists():
            logger = logging.getLogger("DC2Dataset")
            warning_msg = "WARNING: cached data already exists at [%s], we directly use it\n"
            logger.warning(warning_msg, str(self.data_dir))
            return None

        logger = logging.getLogger("DC2Dataset")
        warning_msg = "WARNING: can't find cached data, we generate it at [%s]\n"
        logger.warning(warning_msg, str(self.data_dir))
        self.data_dir.mkdir(parents=True)

        n_image = self._load_image_and_bg_files_list()
        generate_cached_data_wrapper = lambda image_index: generate_cached_data(
            image_index,
            self._image_files,
            self._bg_files,
            self.n_bands,
            self.image_lim,
            self.dc2_cat_path,
            self.bands,
            self.n_image_split,
            self.data_dir,
            self.min_flux_for_loss,
            self.data_in_one_cached_file,
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

    def get_plotting_sample(self, image_idx):
        if self._image_files is None or self._bg_files is None:
            self._load_image_and_bg_files_list()

        if image_idx < 0 or image_idx >= len(self._image_files):
            raise IndexError("invalid image_idx")

        result_dict = load_image_and_catalog(
            image_idx,
            self._image_files,
            self._bg_files,
            self.image_lim,
            self.bands,
            self.n_bands,
            self.dc2_cat_path,
            self.min_flux_for_loss,
        )
        return {
            "tile_catalog": result_dict["tile_dict"],
            "image": torch.from_numpy(result_dict["inputs"]["image"]),
            "background": torch.from_numpy(result_dict["inputs"]["bg"]),
            "match_id": result_dict["other_info"]["match_id"],
            "full_catalog": result_dict["other_info"]["full_cat"],
            "wcs": result_dict["other_info"]["wcs"],
            "psf_params": result_dict["inputs"]["psf_params"],
        }


def squeeze_tile_cat(tile_cat):
    # by calling `.data` here I circumevent the requirement of TileCatalogs that the length
    # of the first dimension is the batch size
    tile_dict_copy = copy.copy(tile_cat.data)
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


def load_image_and_catalog(
    image_idx, image_files, bg_files, image_lim, bands, n_bands, dc2_cat_path, min_flux_for_loss
):
    image_list, bg_list, wcs_header_str = read_frame_for_band(
        image_files, bg_files, image_idx, n_bands, image_lim
    )
    wcs = from_wcs_header_str_to_wcs(wcs_header_str)
    image = np.stack(image_list)
    bg = np.stack(bg_list)

    plocs_lim = image[0].shape
    height = plocs_lim[0]
    width = plocs_lim[1]
    full_cat, psf_params, match_id = DC2FullCatalog.from_file(
        dc2_cat_path, wcs, height, width, bands, min_flux_for_loss
    )
    tile_cat = full_cat.to_tile_catalog(4, 5)
    tile_dict = squeeze_tile_cat(tile_cat)
    tile_dict["star_fluxes"] = tile_dict["star_fluxes"].clamp(min=1e-18)
    tile_dict["galaxy_fluxes"] = tile_dict["galaxy_fluxes"].clamp(min=1e-18)
    tile_dict["galaxy_params"][..., 3] = tile_dict["galaxy_params"][..., 3].clamp(min=1e-18)
    tile_dict["galaxy_params"][..., 5] = tile_dict["galaxy_params"][..., 5].clamp(min=1e-18)

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


def generate_cached_data(
    image_index,
    image_files,
    bg_files,
    n_bands,
    image_lim,
    dc2_cat_path,
    bands,
    n_image_split,
    data_dir,
    min_flux_for_loss,
    data_in_one_cached_file,
):
    result_dict = load_image_and_catalog(
        image_index,
        image_files,
        bg_files,
        image_lim,
        bands,
        n_bands,
        dc2_cat_path,
        min_flux_for_loss,
    )

    image = result_dict["inputs"]["image"]
    bg = result_dict["inputs"]["bg"]
    tile_dict = result_dict["tile_dict"]
    wcs_header_str = result_dict["other_info"]["wcs_header_str"]
    psf_params = result_dict["inputs"]["psf_params"]

    # split image
    split_lim = image_lim[0] // n_image_split
    image = torch.from_numpy(image)
    split_image = split_full_image(image, split_lim)
    image_height_pixels = image.shape[1]
    split_image_num_on_height = image_height_pixels // split_lim
    bg = torch.from_numpy(bg)
    split_bg = split_full_image(bg, split_lim)

    # split tile
    tile_split = {}
    param_list = [
        "locs",
        "n_sources",
        "source_type",
        "galaxy_fluxes",
        "galaxy_params",
        "star_fluxes",
        "star_log_fluxes",
        "one_source_mask",
        "two_sources_mask",
        "more_than_two_sources_mask",
    ]
    for i in param_list:
        split1_tile = torch.stack(torch.split(tile_dict[i], split_lim // 4, dim=0))
        split2_tile = torch.stack(torch.split(split1_tile, split_lim // 4, dim=2))
        tile_split[i] = torch.split(split2_tile.flatten(0, 2), split_lim // 4)

    data_split = {
        "tile_catalog": [dict(zip(tile_split, i)) for i in zip(*tile_split.values())],
        "images": split_image,
        "image_height_index": (
            torch.arange(0, len(split_image)) % split_image_num_on_height
        ).tolist(),
        "image_width_index": (
            torch.arange(0, len(split_image)) // split_image_num_on_height
        ).tolist(),
        "background": split_bg,
        "psf_params": [psf_params for _ in range(n_image_split**2)],
    }

    data_splits = split_list(
        [dict(zip(data_split, i)) for i in zip(*data_split.values())],
        sub_list_len=data_in_one_cached_file,
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
        assert data_count < 1e6 and image_index < 1e5, "too many cached data files"
        assert len(tmp_data_cached) < 1e5, "too many cached data in one file"
        cached_data_file_name = (
            f"cached_data_{image_index:04d}_{data_count:05d}_size_{len(tmp_data_cached):04d}.pt"
        )
        with open(
            data_dir / cached_data_file_name,
            "wb",
        ) as cached_data_file:
            torch.save(tmp_data_cached, cached_data_file)
        data_count += 1


class DC2FullCatalog(FullCatalog):
    """Class for the LSST PHOTO Catalog.

    Some resources:
    - https://data.lsstdesc.org/doc/dc2_sim_sky_survey
    - https://arxiv.org/abs/2010.05926
    """

    @classmethod
    def from_file(cls, cat_path, wcs, height, width, band, min_flux_for_loss):
        catalog = pd.read_pickle(cat_path)
        flux_r_band = catalog["flux_r"].values
        catalog = catalog.loc[flux_r_band > min_flux_for_loss]
        objid = torch.tensor(catalog["id"].values)
        match_id = torch.tensor(catalog["match_objectId"].values)
        ra = torch.tensor(catalog["ra"].values).numpy().squeeze()
        dec = torch.tensor(catalog["dec"].values).numpy().squeeze()
        galaxy_bools = torch.tensor((catalog["truth_type"] == 1).values)
        star_bools = torch.tensor((catalog["truth_type"] == 2).values)
        flux_list, psf_params = get_band(band, catalog)

        flux = torch.stack(flux_list).t()

        galaxy_bulge_frac = torch.tensor(catalog["bulge_to_total_ratio_i"].values)
        galaxy_disk_frac = (1 - galaxy_bulge_frac) * galaxy_bools

        position_angle = torch.tensor(catalog["position_angle_true"].values)
        galaxy_beta_radians = (position_angle / 180) * math.pi * galaxy_bools

        galaxy_a_d = torch.tensor(catalog["size_disk_true"].values) / 2
        galaxy_a_b = torch.tensor(catalog["size_bulge_true"].values) / 2
        galaxy_b_d = torch.tensor(catalog["size_minor_disk_true"].values) / 2
        galaxy_b_b = torch.tensor(catalog["size_minor_bulge_true"].values) / 2
        galaxy_disk_q = (galaxy_b_d / galaxy_a_d) * galaxy_bools
        galaxy_bulge_q = (galaxy_b_b / galaxy_a_b) * galaxy_bools

        galaxy_params = torch.stack(
            (
                galaxy_disk_frac,
                galaxy_beta_radians,
                galaxy_disk_q,
                galaxy_a_d * galaxy_bools,
                galaxy_bulge_q,
                galaxy_a_b * galaxy_bools,
            )
        ).t()

        keep = galaxy_bools | star_bools
        source_type = torch.from_numpy(np.array(catalog["truth_type"])[keep])
        source_type[source_type == 2] = 0
        ra = ra[keep]
        dec = dec[keep]
        match_id = match_id[keep]

        star_fluxes = flux[keep]
        star_log_fluxes = flux.log()[keep]
        galaxy_params = galaxy_params[keep]
        galaxy_fluxes = flux[keep]
        objid = objid[keep]

        pt, pr = wcs.all_world2pix(ra, dec, 0)  # convert to pixel coordinates
        plocs = torch.stack((torch.tensor(pr), torch.tensor(pt)), dim=-1)

        plocs = (plocs.reshape(1, plocs.size()[0], 2))[0]

        x0_mask = (plocs[:, 0] > 0) & (plocs[:, 0] < width)
        x1_mask = (plocs[:, 1] > 0) & (plocs[:, 1] < height)
        x_mask = x0_mask * x1_mask

        star_fluxes = star_fluxes[x_mask]
        star_log_fluxes = star_log_fluxes[x_mask]
        galaxy_params = galaxy_params[x_mask]
        galaxy_fluxes = galaxy_fluxes[x_mask]
        objid = objid[x_mask]
        match_id = match_id[x_mask]

        plocs = plocs[x_mask]
        source_type = source_type[x_mask]

        nobj = source_type.shape[0]
        d = {
            "objid": objid.reshape(1, nobj, 1),
            "n_sources": torch.tensor((nobj,)),
            "source_type": source_type.reshape(1, nobj, 1),
            "plocs": plocs.reshape(1, nobj, 2),
            "galaxy_fluxes": galaxy_fluxes.reshape(1, nobj, 6),
            "galaxy_params": galaxy_params.reshape(1, nobj, 6),
            "star_fluxes": star_fluxes.reshape(1, nobj, 6),
            "star_log_fluxes": star_log_fluxes.reshape(1, nobj, 6),
        }

        return cls(height, width, d), torch.stack(psf_params), match_id


def read_frame_for_band(image_files, bg_files, n, n_bands, image_lim):
    image_list = []
    bg_list = []
    wcs_header_str = None
    for b in range(n_bands):
        image_frame = fits.open(image_files[b][n])
        bg_frame = fits.open(bg_files[b][n])
        image_data = image_frame[1].data
        bg_data = bg_frame[0].data

        if wcs_header_str is None:
            wcs_header_str = image_frame[1].header.tostring()

        image_frame.close()
        bg_frame.close()

        image = torch.nan_to_num(torch.from_numpy(image_data)[: image_lim[0], : image_lim[1]])
        bg = torch.from_numpy(bg_data.astype(np.float32)).expand(image_lim[0], image_lim[1])

        image += bg
        image_list.append(image)
        bg_list.append(bg)

    return image_list, bg_list, wcs_header_str


def split_full_image(image, split_lim):
    split1_image = torch.stack(torch.split(image, split_lim, dim=1))
    split2_image = torch.stack(torch.split(split1_image, split_lim, dim=3))
    return list(torch.split(split2_image.flatten(0, 2), 6))


def get_band(band, catalog):
    flux_list = []
    psf_params = []
    for b in band:
        flux_list.append(torch.tensor((catalog["flux_" + b]).values))
        psf_params_name = ["IxxPSF_pixel_", "IyyPSF_pixel_", "IxyPSF_pixel_", "psf_fwhm_"]
        psf_params_band = []
        for i in psf_params_name:
            median_psf = np.nanmedian((catalog[i + b]).values).astype(np.float32)
            psf_params_band.append(torch.tensor(median_psf))
        psf_params.append(torch.stack(psf_params_band).t())

    return flux_list, psf_params
