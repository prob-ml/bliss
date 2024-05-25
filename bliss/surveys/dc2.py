import collections
import math
import multiprocessing
import pathlib
import random

import numpy as np
import pandas as pd
import torch
from astropy.io import fits
from astropy.io.fits import Header
from astropy.wcs import WCS
from einops import rearrange
from torch.utils.data import DataLoader, Dataset

from bliss.catalog import FullCatalog
from bliss.surveys.survey import Survey


def from_wcs_header_str_to_wcs(wcs_header_str: str):
    return WCS(Header.fromstring(wcs_header_str))


def map_nested_dicts(cur_dict, func):
    if isinstance(cur_dict, collections.abc.Mapping):
        return {k: map_nested_dicts(v, func) for k, v in cur_dict.items()}

    return func(cur_dict)


def generate_split_file(image_index, self_copy):
    split_count = 0
    image_list, bg_list, wcs_header_str = read_frame_for_band(
        self_copy["image_files"],
        self_copy["bg_files"],
        image_index,
        self_copy["n_bands"],
        self_copy["image_lim"],
    )
    wcs = from_wcs_header_str_to_wcs(wcs_header_str)
    image = np.stack(image_list)
    bg = np.stack(bg_list)

    plocs_lim = image[0].shape
    height = plocs_lim[0]
    width = plocs_lim[1]
    full_cat, psf_params, _ = Dc2FullCatalog.from_file(
        self_copy["cat_path"], wcs, height, width, self_copy["bands"]
    )
    tile_dict = full_cat.to_tile_catalog(4, 5).get_brightest_sources_per_tile().to_dict()

    for k, v in tile_dict.items():
        if k != "n_sources":
            tile_dict[k] = rearrange(v, "1 h w nh nw -> h w nh nw")
    tile_dict["n_sources"] = rearrange(tile_dict["n_sources"], "1 h w -> h w")

    tile_dict["star_fluxes"] = tile_dict["star_fluxes"].clamp(min=1e-18)
    tile_dict["galaxy_fluxes"] = tile_dict["galaxy_fluxes"].clamp(min=1e-18)
    tile_dict["galaxy_params"][..., 3] = tile_dict["galaxy_params"][..., 3].clamp(min=1e-18)
    tile_dict["galaxy_params"][..., 5] = tile_dict["galaxy_params"][..., 5].clamp(min=1e-18)

    # split image
    split_lim = self_copy["image_lim"][0] // self_copy["n_split"]
    image = torch.from_numpy(image)
    split_image = split_full_image(image, split_lim)
    image_height_pixels = image.shape[1]
    split_image_num_on_height = image_height_pixels // split_lim
    bg = torch.from_numpy(bg)
    split_bg = split_full_image(bg, split_lim)

    tile_split = {}
    param_list = [
        "locs",
        "n_sources",
        "source_type",
        "galaxy_fluxes",
        "galaxy_params",
        "star_fluxes",
        "star_log_fluxes",
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
        "psf_params": [psf_params for _ in range(self_copy["n_split"] ** 2)],
    }

    data_splits = [dict(zip(data_split, i)) for i in zip(*data_split.values())]
    for cur_split in data_splits:  # noqa: WPS426
        assert split_count < 1e6 and image_index < 1e5, "too many splits"
        split_file_name = f"split_image_{image_index:04d}_{split_count:05d}.pt"
        with open(
            self_copy["split_file_path"] / split_file_name,
            "wb",
        ) as split_file:
            cur_split_copy = map_nested_dicts(
                cur_split, lambda x: x.clone() if isinstance(x, torch.Tensor) else x
            )
            cur_split_copy.update(wcs_header_str=wcs_header_str)
            cur_split_copy.update(split_id=split_file_name)
            torch.save(cur_split_copy, split_file)
        split_count += 1


class DC2Dataset(Dataset):
    def __init__(self, split_files_list) -> None:
        super().__init__()
        self.split_files_list = split_files_list

    def __len__(self):
        return len(self.split_files_list)

    def __getitem__(self, idx):
        with open(self.split_files_list[idx], "rb") as split_file:
            split = torch.load(split_file)
        return split


class DC2(Survey):
    # why are these bands out of order? why does a test break if they are ordered correctly?
    BANDS = ("g", "i", "r", "u", "y", "z")

    def __init__(
        self,
        data_dir,
        cat_path,
        batch_size,
        n_split,
        image_lim,
        num_workers,
        split_result_folder,
        split_processes_num,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.cat_path = cat_path
        self.batch_size = batch_size
        self.bands = self.BANDS
        self.n_bands = len(self.BANDS)
        self.split_files_list = None
        self.train_dataset = None
        self.valid_dataset = None
        self.test_dataset = None
        self.total_dataset = None
        self.n_split = n_split
        self.image_lim = image_lim
        self.num_workers = num_workers
        self.split_file_path = pathlib.Path(data_dir) / split_result_folder
        self.split_processes_num = split_processes_num

        self.image_files = None
        self.bg_files = None

        self._predict_batch = None
        self._image_ids = None

    def __len__(self):
        return len(self.split_files_list)

    def __getitem__(self, idx):
        return self.total_dataset[idx]

    def get_plotting_sample(self, image_idx):
        if self.image_files is None or self.bg_files is None:
            self._load_image_and_bg_files_list()

        image_list, bg_list, wcs_header_str = read_frame_for_band(
            self.image_files, self.bg_files, image_idx, self.n_bands, self.image_lim
        )
        wcs = from_wcs_header_str_to_wcs(wcs_header_str)
        image = np.stack(image_list)
        bg = np.stack(bg_list)

        plocs_lim = image[0].shape
        height = plocs_lim[0]
        width = plocs_lim[1]
        full_cat, psf_params, match_id = Dc2FullCatalog.from_file(
            self.cat_path, wcs, height, width, self.bands
        )
        tile_dict = full_cat.to_tile_catalog(4, 5).get_brightest_sources_per_tile().to_dict()

        for k, v in tile_dict.items():
            if k != "n_sources":
                tile_dict[k] = rearrange(v, "1 h w nh nw -> h w nh nw")
        tile_dict["n_sources"] = rearrange(tile_dict["n_sources"], "1 h w -> h w")

        tile_dict["star_fluxes"] = tile_dict["star_fluxes"].clamp(min=1e-18)
        tile_dict["galaxy_fluxes"] = tile_dict["galaxy_fluxes"].clamp(min=1e-18)
        tile_dict["galaxy_params"][..., 3] = tile_dict["galaxy_params"][..., 3].clamp(min=1e-18)
        tile_dict["galaxy_params"][..., 5] = tile_dict["galaxy_params"][..., 5].clamp(min=1e-18)

        return {
            "tile_catalog": tile_dict,
            "image": torch.from_numpy(image),
            "background": torch.from_numpy(bg),
            "match_id": match_id,
            "full_catalog": full_cat,
            "wcs": wcs,
            "psf_params": psf_params,
        }

    def image_id(self, idx: int):
        return self.split_files_list[idx].name

    def idx(self, image_id: str) -> int:
        if self._image_ids is None:
            self._image_ids = [split_file.name for split_file in self.split_files_list]
        return self._image_ids.index(image_id)

    def image_ids(self):
        if self._image_ids is None:
            self._image_ids = [split_file.name for split_file in self.split_files_list]
        return self._image_ids

    def _load_image_and_bg_files_list(self):
        img_pattern = "**/*/calexp*.fits"
        bg_pattern = "**/*/bkgd*.fits"
        image_files = []
        bg_files = []

        for band in self.bands:
            band_path = self.data_dir + str(band)
            img_file_list = list(pathlib.Path(band_path).glob(img_pattern))
            bg_file_list = list(pathlib.Path(band_path).glob(bg_pattern))

            image_files.append(sorted(img_file_list))
            bg_files.append(sorted(bg_file_list))
        n_image = len(bg_files[0])

        # assign state only in main process
        self.image_files = image_files
        self.bg_files = bg_files

        return n_image

    def prepare_data(self):  # noqa: WPS324
        if self.split_file_path.exists():
            return None
        self.split_file_path.mkdir(parents=True)

        n_image = self._load_image_and_bg_files_list()

        with multiprocessing.Pool(processes=self.split_processes_num) as process_pool:
            process_pool.starmap(
                generate_split_file,
                zip(
                    list(range(n_image)),
                    [
                        {
                            "image_files": self.image_files,
                            "bg_files": self.bg_files,
                            "n_bands": self.n_bands,
                            "image_lim": self.image_lim,
                            "cat_path": self.cat_path,
                            "bands": self.bands,
                            "n_split": self.n_split,
                            "split_file_path": self.split_file_path,
                        }
                        for _ in range(n_image)
                    ],
                ),
                chunksize=4,
            )

        return None

    def setup(self, stage="fit"):
        self.split_files_list = list(self.split_file_path.glob("split_*.pt"))
        random.Random(218).shuffle(self.split_files_list)

        data_len = len(self.split_files_list)
        train_len = int(data_len * 0.8)
        val_len = int(data_len * 0.1)

        self.train_dataset = DC2Dataset(self.split_files_list[:train_len])
        self.valid_dataset = DC2Dataset(self.split_files_list[train_len : (train_len + val_len)])
        self.test_dataset = DC2Dataset(self.split_files_list[(train_len + val_len) :])
        self.total_dataset = DC2Dataset(self.split_files_list)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            shuffle=True,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid_dataset, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers // 2
        )


class Dc2FullCatalog(FullCatalog):
    """Class for the LSST PHOTO Catalog.

    Some resources:
    - https://data.lsstdesc.org/doc/dc2_sim_sky_survey
    - https://arxiv.org/abs/2010.05926
    """

    @classmethod
    def from_file(cls, cat_path, wcs, height, width, band):
        catalog = pd.read_pickle(cat_path)
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
        image_data = image_frame[1].data  # pylint: disable=maybe-no-member
        bg_data = bg_frame[0].data  # pylint: disable=maybe-no-member

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
