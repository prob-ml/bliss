import glob
import math
import random
import warnings

import numpy as np
import pandas as pd
import torch
from astropy.io import fits
from astropy.wcs import WCS, FITSFixedWarning
from einops import rearrange
from torch.utils.data import DataLoader

from bliss.catalog import FullCatalog
from bliss.surveys.survey import Survey


class DC2(Survey):
    BANDS = "r"

    def __init__(self, data_dir, cat_path):
        super().__init__()
        self.data_dir = data_dir
        self.cat_path = cat_path
        self.band = self.BANDS
        self.dc2_data = []
        self.data = []
        self.valid = []
        self.test = []
        self.prepare_data()
        self.setup()

        self._predict_batch = {
            "images": self.dc2_data[0]["images"],
            "background": self.dc2_data[0]["background"],
        }

    def __len__(self):
        return len(self.dc2_data)

    def __getitem__(self, idx):
        return self.dc2_data(idx)

    def image_id(self, idx: int):
        return self.dc2_data[idx]["images"]

    def idx(self, image_id: int) -> int:
        return self.dc2_data[image_id]

    def image_ids(self):
        return [self.dc2_data[i]["images"] for i in range(len(self))]

    def prepare_data(self):
        bg_files = glob.glob(self.data_dir + "bkgd" + "*.fits")
        image_files = glob.glob(self.data_dir + "calexp" + "*.fits")

        bg_files = sorted(bg_files)
        image_files = sorted(image_files)
        n_image = len(bg_files)

        data = []

        for i in range(n_image):
            image_frame = fits.open(image_files[i])
            bg_frame = fits.open(bg_files[i])
            image_data = image_frame[1].data  # pylint: disable=maybe-no-member
            bg_data = bg_frame[0].data  # pylint: disable=maybe-no-member
            image = image_data[:, 36:-36].astype(np.float32)
            bg = np.repeat(np.repeat(bg_data, 128, axis=1), 128, axis=0)
            bg_crop = bg[48:-48, 48:-48].astype(np.float32)
            image += bg_crop

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=FITSFixedWarning)
                wcs = WCS(image_frame[0])

            image_frame.close()
            bg_frame.close()

            plocs_lim = image.shape
            height = plocs_lim[0]
            width = plocs_lim[1]
            full_cat = Dc2FullCatalog.from_file(self.cat_path, wcs, height, width, self.band)
            tile_cat = full_cat.to_tile_params(4, 2)
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

            # split image to 25
            split_h = np.linspace(0, plocs_lim[0], num=5, endpoint=False)
            split_w = np.linspace(0, plocs_lim[1], num=5, endpoint=False)
            image_h = plocs_lim[0] // 5
            image_w = plocs_lim[1] // 5

            for i in split_h:
                for j in split_w:
                    image1 = image[int(i) : int(i) + image_h, int(j) : int(j) + image_w]
                    bg1 = bg_crop[int(i) : int(i) + image_h, int(j) : int(j) + image_w]

                    tlim = [
                        int(i // 4),
                        int((i + image_h) // 4),
                        int(j // 4),
                        int((j + image_w) // 4),
                    ]
                    tile1 = get_tile((tlim[0], tlim[1], tlim[2], tlim[3]), tile_dict)

                    data.append(
                        {
                            "tile_catalog": tile1,
                            "images": rearrange(torch.from_numpy(image1), "h w -> 1 h w"),
                            "background": rearrange(torch.from_numpy(bg1), "h w -> 1 h w"),
                        }
                    )

        random.shuffle(data)
        self.dc2_data = data

    @property
    def predict_batch(self):
        if not self._predict_batch:
            self._predict_batch = {
                "images": self.dc2_data[0]["image"],
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
        return DataLoader(self.data, batch_size=1)

    def val_dataloader(self):
        return DataLoader(self.valid, batch_size=1)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=1)


class Dc2FullCatalog(FullCatalog):
    """Class for the LSST PHOTO Catalog.

    Some resources:
    - https://data.lsstdesc.org/doc/dc2_sim_sky_survey
    - https://arxiv.org/abs/2010.05926
    """

    @classmethod
    def from_file(cls, cat_path, wcs, height, width, band):
        data = pd.read_pickle(cat_path + band + ".pkl")
        ra = torch.tensor(data["ra"].values)
        dec = torch.tensor(data["dec"].values)
        galaxy_bools = torch.tensor((data["truth_type"] == 1).values)
        star_bools = torch.tensor((data["truth_type"] == 2).values)
        flux = torch.tensor((data["flux_" + band]).values)

        galaxy_bulge_frac = torch.tensor(data["bulge_to_total_ratio_i"].values)
        galaxy_disk_frac = (1 - galaxy_bulge_frac) * galaxy_bools

        position_angle = torch.tensor(data["position_angle_true"].values)
        galaxy_beta_radians = (position_angle / 180) * math.pi * galaxy_bools

        galaxy_a_d = torch.tensor(data["size_disk_true"].values) / 2
        galaxy_a_b = torch.tensor(data["size_bulge_true"].values) / 2
        galaxy_b_d = torch.tensor(data["size_minor_disk_true"].values) / 2
        galaxy_b_b = torch.tensor(data["size_minor_bulge_true"].values) / 2
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
        source_type = torch.from_numpy(np.array(data["truth_type"])[keep])
        source_type[source_type == 2] = 0
        ra = ra[keep]
        dec = dec[keep]

        star_fluxes = flux[keep]
        star_log_fluxes = flux.log()[keep]
        galaxy_params = galaxy_params[keep]
        galaxy_fluxes = flux[keep]

        pt, pr = wcs.all_world2pix(ra, dec, 0)  # convert to pixel coordinates
        pt = torch.tensor(pt)
        pr = torch.tensor(pr)
        plocs = torch.stack((pr, pt), dim=-1)
        plocs[:, 0] -= 36

        plocs = (plocs.reshape(1, plocs.size()[0], 2))[0]

        x0_mask = (plocs[:, 0] > 0) & (plocs[:, 0] < width)
        x1_mask = (plocs[:, 1] > 0) & (plocs[:, 1] < height)
        x_mask = x0_mask * x1_mask

        star_fluxes = star_fluxes[x_mask]
        star_log_fluxes = star_log_fluxes[x_mask]
        galaxy_params = galaxy_params[x_mask]
        galaxy_fluxes = galaxy_fluxes[x_mask]

        plocs = plocs[x_mask]
        plocs_rv = plocs.clone()
        plocs_rv[:, 0] = plocs[:, 1]
        plocs_rv[:, 1] = plocs[:, 0]
        source_type = source_type[x_mask]

        nobj = source_type.shape[0]
        d = {
            "n_sources": torch.tensor((nobj,)),
            "source_type": source_type.reshape(1, nobj, 1),
            "plocs": plocs_rv.reshape(1, nobj, 2),
            "galaxy_fluxes": galaxy_fluxes.reshape(1, nobj, 1),
            "galaxy_params": galaxy_params.reshape(1, nobj, 6),
            "star_fluxes": star_fluxes.reshape(1, nobj, 1),
            "star_log_fluxes": star_log_fluxes.reshape(1, nobj, 1),
        }

        return cls(height, width, d)


def get_tile(split_lim, tile_cat):
    tile_dict = {}
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
        tile_dict[i] = tile_cat[i][split_lim[0] : split_lim[1], split_lim[2] : split_lim[3]]
    return tile_dict
