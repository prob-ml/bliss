import logging
import math
import sys
from typing import List

import pandas as pd
import torch

from bliss.catalog import BaseTileCatalog
from bliss.surveys.dc2 import (
    DC2DataModule,
    DC2FullCatalog,
    map_nested_dicts,
    unpack_dict,
    wcs_from_wcs_header_str,
)
from case_studies.weak_lensing.utils.weighted_avg_ellip import compute_weighted_avg_ellip


class LensingDC2DataModule(DC2DataModule):
    def __init__(
        self,
        dc2_image_dir: str,
        dc2_cat_path: str,
        image_slen: int,  # assume square images: image_slen x image_slen
        n_image_split: int,
        tile_slen: int,
        splits: str,
        avg_ellip_kernel_size: int,
        avg_ellip_kernel_sigma: int,
        batch_size: int,
        num_workers: int,
        cached_data_path: str,
        train_transforms: List,
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
            catalog_min_r_flux=-sys.maxsize - 1,  # smaller than any int
            prepare_data_processes_num=1,
            data_in_one_cached_file=100000,
            splits=splits,
            batch_size=batch_size,
            num_workers=num_workers,
            cached_data_path=cached_data_path,
            train_transforms=train_transforms,
            nontrain_transforms=[],
            subset_fraction=None,
        )

        self.image_slen = image_slen
        self.bands = self.BANDS
        self.n_bands = len(self.BANDS)
        self.avg_ellip_kernel_size = avg_ellip_kernel_size
        self.avg_ellip_kernel_sigma = avg_ellip_kernel_sigma

    # override prepare_data
    def prepare_data(self):
        if self.cached_data_path.exists():
            logger = logging.getLogger("LensingDC2DataModule")
            warning_msg = "WARNING: cached data already exists at [%s], we directly use it\n"
            logger.warning(warning_msg, str(self.cached_data_path))
            return

        logger = logging.getLogger("LensingDC2DataModule")
        warning_msg = "WARNING: can't find cached data, we generate it at [%s]\n"
        logger.warning(warning_msg, str(self.cached_data_path))
        self.cached_data_path.mkdir(parents=True)

        n_image = self._load_image_and_bg_files_list()

        for i in range(n_image):
            self.generate_cached_data(i)

    def to_tile_catalog(self, full_catalog, height, width):
        plocs = full_catalog["plocs"].reshape(1, -1, 2)
        source_tile_coords = torch.div(plocs, self.tile_slen, rounding_mode="trunc").to(torch.int64)
        n_tiles_h = math.ceil(height / self.tile_slen)
        n_tiles_w = math.ceil(width / self.tile_slen)
        stti = source_tile_coords[:, :, 0] * n_tiles_w + source_tile_coords[:, :, 1]
        source_to_tile_indices_default = stti.unsqueeze(-1).to(dtype=torch.int64)

        tile_cat = {}

        num_tiles = n_tiles_h * n_tiles_w

        for k, v in full_catalog.items():
            if k == "plocs":
                continue

            v = v.reshape(self.batch_size, plocs.shape[1], -1)
            if k == "mag_mask":
                continue

            v_sum = torch.zeros(self.batch_size, num_tiles, v.shape[-1], dtype=v.dtype)
            v_count = torch.zeros(self.batch_size, num_tiles, v.shape[-1], dtype=v.dtype)
            add_pos = torch.ones_like(v)

            if k == "psf":
                psf_nanmask = ~torch.isnan(v).any(dim=-1, keepdim=True).expand(-1, -1, v.shape[-1])
                v = torch.where(psf_nanmask, v, torch.tensor(0.0))
                add_pos = psf_nanmask.float().to(v.dtype)
                source_to_tile_indices = source_to_tile_indices_default.expand(-1, -1, v.shape[-1])
            else:
                source_to_tile_indices = source_to_tile_indices_default
            v_sum = v_sum.scatter_add(1, source_to_tile_indices, v)
            v_count = v_count.scatter_add(1, source_to_tile_indices, add_pos)
            tile_cat[k + "_sum"] = v_sum.reshape(self.batch_size, n_tiles_w, n_tiles_h, v.shape[-1])
            tile_cat[k + "_count"] = v_count.reshape(
                self.batch_size, n_tiles_w, n_tiles_h, v.shape[-1]
            )
        return BaseTileCatalog(tile_cat)

    # override load_image_and_catalog
    def load_image_and_catalog(self, image_index):
        image, wcs_header_str = self.read_image_for_bands(image_index)
        wcs = wcs_from_wcs_header_str(wcs_header_str)

        plocs_lim = image[0].shape
        height = plocs_lim[0]
        width = plocs_lim[1]
        full_cat, psf_params = LensingDC2Catalog.from_file(
            self.dc2_cat_path,
            wcs,
            height,
            width,
            bands=self.bands,
            n_bands=self.n_bands,
        )

        tile_cat = self.to_tile_catalog(full_cat, height, width)
        psf_params = tile_cat["psf_sum"] / tile_cat["psf_count"]
        del tile_cat["psf_sum"]
        del tile_cat["psf_count"]
        tile_dict = self.squeeze_tile_dict(tile_cat.data)

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
            },
        }

    # override generate_cached_data
    def generate_cached_data(self, image_index):
        result_dict = self.load_image_and_catalog(image_index)

        image = result_dict["inputs"]["image"]
        tile_dict = result_dict["tile_dict"]
        psf_params = result_dict["inputs"]["psf_params"]

        shear1 = tile_dict["shear1_sum"] / tile_dict["shear1_count"]
        shear2 = tile_dict["shear2_sum"] / tile_dict["shear2_count"]
        convergence = tile_dict["convergence_sum"] / tile_dict["convergence_count"]
        ellip1_lensed = tile_dict["ellip1_lensed_sum"] / tile_dict["ellip1_lensed_count"]
        ellip2_lensed = tile_dict["ellip2_lensed_sum"] / tile_dict["ellip2_lensed_count"]
        ellip_lensed = torch.stack((ellip1_lensed.squeeze(-1), ellip2_lensed.squeeze(-1)), dim=-1)
        redshift = tile_dict["redshift_sum"] / tile_dict["redshift_count"]
        ra = tile_dict["ra_sum"] / tile_dict["ra_count"]
        dec = tile_dict["dec_sum"] / tile_dict["dec_count"]

        tile_dict["shear_1"] = shear1
        tile_dict["shear_2"] = shear2
        tile_dict["convergence"] = convergence
        tile_dict["ellip_lensed"] = ellip_lensed
        tile_dict["ellip_lensed_wavg"] = compute_weighted_avg_ellip(
            tile_dict, self.avg_ellip_kernel_size, self.avg_ellip_kernel_sigma
        )
        tile_dict["redshift"] = redshift
        tile_dict["ra"] = ra
        tile_dict["dec"] = dec

        data_splits = self.split_image_and_tile_cat(image, tile_dict, tile_dict.keys(), psf_params)

        data_to_cache = unpack_dict(data_splits)

        for i in range(self.n_image_split**2):  # noqa: WPS426
            cached_data_file_name = f"cached_data_{image_index:04d}_{i:04d}_size_1.pt"
            tmp = data_to_cache[i]
            tmp_clone = map_nested_dicts(
                tmp, lambda x: x.clone() if isinstance(x, torch.Tensor) else x
            )
            with open(self.cached_data_path / cached_data_file_name, "wb") as cached_data_file:
                torch.save([tmp_clone], cached_data_file)


class LensingDC2Catalog(DC2FullCatalog):
    @classmethod
    def from_file(cls, cat_path, wcs, height, width, **kwargs):
        catalog = pd.read_pickle(cat_path)

        galid = torch.from_numpy(catalog["galaxy_id"].values)
        ra = torch.from_numpy(catalog["ra"].values)
        dec = torch.from_numpy(catalog["dec"].values)

        shear1 = torch.from_numpy(catalog["shear_1"].values)
        shear2 = torch.from_numpy(catalog["shear_2"].values)
        complex_shear = shear1 + shear2 * 1j
        convergence = torch.from_numpy(catalog["convergence"].values)
        reduced_shear = complex_shear / (1.0 - convergence)

        ellip1_intrinsic = torch.from_numpy(catalog["ellipticity_1_true"].values)
        ellip2_intrinsic = torch.from_numpy(catalog["ellipticity_2_true"].values)
        complex_ellip_intrinsic = ellip1_intrinsic + ellip2_intrinsic * 1j
        complex_ellip_lensed = (complex_ellip_intrinsic + reduced_shear) / (
            1.0 + reduced_shear.conj() * complex_ellip_intrinsic
        )
        ellip1_lensed = torch.view_as_real(complex_ellip_lensed)[..., 0]
        ellip2_lensed = torch.view_as_real(complex_ellip_lensed)[..., 1]

        redshift = torch.from_numpy(catalog["redshift"].values)

        _, psf_params = cls.get_bands_flux_and_psf(kwargs["bands"], catalog, median=False)
        # psf_params is n_bands x 4 (n_params) x n_measures

        plocs = cls.plocs_from_ra_dec(ra, dec, wcs).squeeze(0)
        x0_mask = (plocs[:, 0] > 0) & (plocs[:, 0] < height)
        x1_mask = (plocs[:, 1] > 0) & (plocs[:, 1] < width)
        plocs_mask = x0_mask * x1_mask

        galid = galid[plocs_mask]
        ra = ra[plocs_mask]
        dec = dec[plocs_mask]
        plocs = plocs[plocs_mask]

        shear1 = shear1[plocs_mask]
        shear2 = shear2[plocs_mask]
        convergence = convergence[plocs_mask]
        ellip1_lensed = ellip1_lensed[plocs_mask]
        ellip2_lensed = ellip2_lensed[plocs_mask]

        redshift = redshift[plocs_mask]

        psf_params = psf_params[:, :, :, plocs_mask.squeeze() == 1]
        psf_params = psf_params.permute(0, 3, 1, 2).flatten(2, -1)  # 1, n_obj, 24

        nobj = galid.shape[0]

        d = {
            "ra": ra.reshape(1, nobj, 1),
            "dec": dec.reshape(1, nobj, 1),
            "plocs": plocs.reshape(1, nobj, 2),
            "shear1": shear1.reshape(1, nobj, 1),
            "shear2": shear2.reshape(1, nobj, 1),
            "convergence": convergence.reshape(1, nobj, 1),
            "ellip1_lensed": ellip1_lensed.reshape(1, nobj, 1),
            "ellip2_lensed": ellip2_lensed.reshape(1, nobj, 1),
            "redshift": redshift.reshape(1, nobj, 1),
            "psf": psf_params.reshape(1, nobj, -1),
        }

        return cls(height, width, d), psf_params
