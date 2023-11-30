import math
import pathlib
import random

import numpy as np
import pandas as pd
import torch
from astropy.io import fits
from astropy.wcs import WCS
from einops import rearrange
from torch.utils.data import DataLoader

from bliss.catalog import FullCatalog
from bliss.surveys.survey import Survey


class DC2(Survey):
    # why are these bands out of order? why does a test break if they are ordered correctly?
    BANDS = ("g", "i", "r", "u", "y", "z")

    def __init__(
        self, data_dir, cat_path, batch_size, n_split, image_lim, use_deconv_channel, deconv_path
    ):
        super().__init__()
        self.data_dir = data_dir
        self.cat_path = cat_path
        self.batch_size = batch_size
        self.bands = self.BANDS
        self.n_bands = len(self.BANDS)
        self.dc2_data = []
        self.data = []
        self.valid = []
        self.test = []
        self.n_split = n_split
        self.image_lim = image_lim
        self.use_deconv_channel = use_deconv_channel
        self.deconv_path = deconv_path

        self._predict_batch = None

    def __len__(self):
        return len(self.dc2_data)

    def __getitem__(self, idx):
        return self.dc2_data[idx]

    def image_id(self, idx: int):
        return self.dc2_data[idx]["images"]

    def idx(self, image_id: int) -> int:
        return self[image_id]

    def image_ids(self):
        return [self.dc2_data[i]["images"] for i in range(len(self))]

    def prepare_data(self):
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

        data = []

        for n in range(n_image):
            image_list, bg_list, wcs = read_frame_for_band(
                image_files, bg_files, n, self.n_bands, self.image_lim
            )
            image = np.stack(image_list)
            bg = np.stack(bg_list)

            plocs_lim = image[0].shape
            height = plocs_lim[0]
            width = plocs_lim[1]
            full_cat, psf_params = Dc2FullCatalog.from_file(
                self.cat_path, wcs, height, width, self.bands
            )
            tile_cat = full_cat.to_tile_catalog(4, 5).get_brightest_sources_per_tile()
            tile_dict = tile_cat.to_dict()

            tile_dict["locs"] = rearrange(tile_cat.to_dict()["locs"], "1 h w nh nw -> h w nh nw")
            tile_dict["n_sources"] = rearrange(tile_cat.to_dict()["n_sources"], "1 h w -> h w")
            tile_dict["source_type"] = rearrange(
                tile_cat.to_dict()["source_type"], "1 h w nh nw -> h w nh nw"
            )
            tile_dict["galaxy_fluxes"] = rearrange(
                tile_cat.to_dict()["galaxy_fluxes"], "1 h w nh nw -> h w nh nw"
            )
            tile_dict["galaxy_params"] = rearrange(
                tile_cat.to_dict()["galaxy_params"], "1 h w nh nw -> h w nh nw"
            )
            tile_dict["star_fluxes"] = rearrange(
                tile_cat.to_dict()["star_fluxes"], "1 h w nh nw -> h w nh nw"
            )
            tile_dict["star_log_fluxes"] = rearrange(
                tile_cat.to_dict()["star_log_fluxes"], "1 h w nh nw -> h w nh nw"
            )

            tile_dict["star_fluxes"] = tile_dict["star_fluxes"].clamp(min=1e-18)
            tile_dict["galaxy_fluxes"] = tile_dict["galaxy_fluxes"].clamp(min=1e-18)
            tile_dict["galaxy_params"][..., 3] = tile_dict["galaxy_params"][..., 3].clamp(min=1e-18)
            tile_dict["galaxy_params"][..., 5] = tile_dict["galaxy_params"][..., 5].clamp(min=1e-18)

            # split image
            split_lim = self.image_lim[0] // self.n_split
            image = torch.from_numpy(image)
            split_image = split_full_image(image, split_lim)
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
                "background": split_bg,
                "psf_params": [psf_params for _ in range(self.n_split**2)],
            }

            if self.use_deconv_channel:
                file_path = self.deconv_path + "/" + str(image_files[0][n])[-15:-5] + ".pt"
                deconv_images = torch.load(file_path)
                data_split["deconvolution"] = split_full_image(deconv_images, split_lim)

            data.extend([dict(zip(data_split, i)) for i in zip(*data_split.values())])

        random.shuffle(data)
        self.dc2_data = data

    @property
    def predict_batch(self):
        if not self._predict_batch:
            self._predict_batch = {
                "images": self.dc2_data[0]["images"],
                "background": self.dc2_data[0]["background"],
            }
        return self._predict_batch

    @predict_batch.setter
    def predict_batch(self, value):
        self._predict_batch = value

    def setup(self, stage="fit"):
        data_len = len(self.dc2_data)
        train_len = int(data_len * 0.8)
        val_len = int(data_len * 0.1)

        self.data = self.dc2_data[:train_len]
        self.valid = self.dc2_data[train_len : train_len + val_len]
        self.test = self.dc2_data[train_len + val_len :]

    def train_dataloader(self):
        return DataLoader(self.data, shuffle=True, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.valid, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size)


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
        ra = torch.tensor(catalog["ra"].values)
        dec = torch.tensor(catalog["dec"].values)
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

        ra = ra.numpy().squeeze()
        dec = dec.numpy().squeeze()

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

        star_fluxes = flux[keep]
        star_log_fluxes = flux.log()[keep]
        galaxy_params = galaxy_params[keep]
        galaxy_fluxes = flux[keep]
        objid = objid[keep]

        pt, pr = wcs.all_world2pix(ra, dec, 0)  # convert to pixel coordinates
        pt = torch.tensor(pt)
        pr = torch.tensor(pr)
        plocs = torch.stack((pr, pt), dim=-1)

        plocs = (plocs.reshape(1, plocs.size()[0], 2))[0]

        x0_mask = (plocs[:, 0] > 0) & (plocs[:, 0] < width)
        x1_mask = (plocs[:, 1] > 0) & (plocs[:, 1] < height)
        x_mask = x0_mask * x1_mask

        star_fluxes = star_fluxes[x_mask]
        star_log_fluxes = star_log_fluxes[x_mask]
        galaxy_params = galaxy_params[x_mask]
        galaxy_fluxes = galaxy_fluxes[x_mask]
        objid = objid[x_mask]

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

        return cls(height, width, d), torch.stack(psf_params)


def read_frame_for_band(image_files, bg_files, n, n_bands, image_lim):
    image_list = []
    bg_list = []
    wcs = None
    for b in range(n_bands):
        image_frame = fits.open(image_files[b][n])
        bg_frame = fits.open(bg_files[b][n])
        image_data = image_frame[1].data  # pylint: disable=maybe-no-member
        bg_data = bg_frame[0].data  # pylint: disable=maybe-no-member

        if wcs is None:
            wcs = WCS(image_frame[1].header)  # pylint: disable=maybe-no-member

        image_frame.close()
        bg_frame.close()

        image = torch.nan_to_num(torch.from_numpy(image_data)[: image_lim[0], : image_lim[1]])
        bg = torch.from_numpy(bg_data.astype(np.float32)).expand(image_lim[0], image_lim[1])

        image += bg
        image_list.append(image)
        bg_list.append(bg)

    return image_list, bg_list, wcs


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
