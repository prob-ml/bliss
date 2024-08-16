import logging
import math
import sys

import pandas as pd
import torch

from bliss.catalog import BaseTileCatalog
from bliss.surveys.dc2 import DC2DataModule, DC2FullCatalog, wcs_from_wcs_header_str


class LensingDC2DataModule(DC2DataModule):
    def __init__(
        self,
        dc2_image_dir: str,
        dc2_cat_path: str,
        image_slen: int,  # assume square images: image_slen x image_slen
        tile_slen: int,
        splits: str,
        batch_size: int,
        num_workers: int,
        cached_data_path: str,
        mag_max_cut: float = None,
        **kwargs,
    ):
        super().__init__(
            dc2_image_dir=dc2_image_dir,
            dc2_cat_path=dc2_cat_path,
            image_lim=[image_slen, image_slen],
            n_image_split=1,
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
            train_transforms=[],
            nontrain_transforms=[],
            subset_fraction=None,
        )

        self.image_slen = image_slen
        self.bands = self.BANDS
        self.n_bands = len(self.BANDS)
        self.mag_max_cut = mag_max_cut

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

        generate_data_input = {
            "tile_slen": self.tile_slen,
            "bands": self.bands,
            "n_bands": self.n_bands,
            "dc2_cat_path": self.dc2_cat_path,
            "cached_data_path": self.cached_data_path,
            "mag_max_cut": self.mag_max_cut,
        }

        generate_data_wrapper = lambda image_index: self.generate_cached_data(
            image_index,
            **generate_data_input,
        )

        for i in range(n_image):
            generate_data_wrapper(i)

    def to_tile_catalog(self, full_catalog, height, width):
        plocs = full_catalog["plocs"].reshape(1, -1, 2)
        source_tile_coords = torch.div(plocs, self.tile_slen, rounding_mode="trunc").to(torch.int64)
        n_tiles_h = math.ceil(height / self.tile_slen)
        n_tiles_w = math.ceil(width / self.tile_slen)
        stti = source_tile_coords[:, :, 0] * n_tiles_w + source_tile_coords[:, :, 1]
        source_to_tile_indices = stti.unsqueeze(-1).to(dtype=torch.int64)

        tile_cat = {}

        num_tiles = n_tiles_h * n_tiles_w

        for k, v in full_catalog.items():
            if k == "plocs":
                continue

            v = v.reshape(self.batch_size, plocs.shape[1], 1)
            if k == "mag_mask":
                continue

            v_sum = torch.zeros(self.batch_size, num_tiles, 1, dtype=v.dtype)
            v_count = torch.zeros(self.batch_size, num_tiles, 1, dtype=v.dtype)
            add_pos = torch.ones_like(v)

            if k in set("ellip1_lensed", "ellip2_lensed"):
                mag_mask = full_catalog["magnitude_cut_mask"]
                v = torch.where(mag_mask, v, torch.tensor(0.0))
                add_pos = mag_mask.float().to(v.dtype)

            v_sum = v_sum.scatter_add(1, source_to_tile_indices, v)
            v_count = v_count.scatter_add(1, source_to_tile_indices, add_pos)
            tile_cat[k + "_sum"] = v_sum.reshape(self.batch_size, n_tiles_w, n_tiles_h, 1)
            tile_cat[k + "_count"] = v_count.reshape(self.batch_size, n_tiles_w, n_tiles_h, 1)
        return BaseTileCatalog(tile_cat)

    def load_image_and_catalog(self, image_index, **kwargs):
        image, wcs_header_str = self.read_image_for_bands(image_index)
        wcs = wcs_from_wcs_header_str(wcs_header_str)

        plocs_lim = image[0].shape
        height = plocs_lim[0]
        width = plocs_lim[1]
        full_cat, psf_params = LensingDC2Catalog.from_file(
            kwargs["dc2_cat_path"],
            wcs,
            height,
            width,
            mag_max_cut=kwargs["mag_max_cut"],
            bands=kwargs["bands"],
            n_bands=kwargs["n_bands"],
        )

        tile_cat = self.to_tile_catalog(full_cat, height, width)

        return {
            "tile_dict": tile_cat,
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

    def generate_cached_data(self, image_index, **kwargs):
        result_dict = self.load_image_and_catalog(image_index, **kwargs)

        image = result_dict["inputs"]["image"]
        tile_dict = result_dict["tile_dict"]
        wcs_header_str = result_dict["other_info"]["wcs_header_str"]
        psf_params = result_dict["inputs"]["psf_params"]

        shear1 = (tile_dict["shear1_sum"] / tile_dict["shear1_count"]) * 100
        shear2 = (tile_dict["shear2_sum"] / tile_dict["shear2_count"]) * 100
        shear = torch.stack((shear1.squeeze(-1), shear2.squeeze(-1)), dim=-1)
        convergence = (tile_dict["convergence_sum"] / tile_dict["convergence_count"]) * 100
        ellip1_lensed = (tile_dict["ellip1_lensed_sum"] / tile_dict["ellip1_lensed_count"]) * 100
        ellip2_lensed = (tile_dict["ellip2_lensed_sum"] / tile_dict["ellip2_lensed_count"]) * 100
        ellip_lensed = torch.stack((ellip1_lensed.squeeze(-1), ellip2_lensed.squeeze(-1)), dim=-1)
        redshift = tile_dict["redshift_sum"] / tile_dict["redshift_count"]

        tile_dict["shear"] = shear
        tile_dict["convergence"] = convergence
        tile_dict["ellip_lensed"] = ellip_lensed
        tile_dict["redshift"] = redshift
        tile_dict["tile_size"] = torch.ones_like(convergence) * self.tile_slen

        data_to_cache = {
            "tile_catalog": tile_dict,
            "images": image,
            "psf_params": psf_params,
            "wcs_header_str": wcs_header_str,
        }

        # Create file name for cached data
        cached_data_file_name = f"cached_data_{image_index:04d}.pt"

        # Save all data to a single file (no splits)
        with open(kwargs["cached_data_path"] / cached_data_file_name, "wb") as cached_data_file:
            torch.save(data_to_cache, cached_data_file)


class LensingDC2Catalog(DC2FullCatalog):
    @classmethod
    def from_file(cls, cat_path, wcs, height, width, mag_max_cut=None, **kwargs):
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

        if mag_max_cut:
            mag_true_r = torch.from_numpy(catalog["mag_true_r"].values)
            mag_mask = mag_true_r < mag_max_cut
        else:
            mag_mask = torch.ones_like(galid).bool()

        _, psf_params = cls.get_bands_flux_and_psf(kwargs["bands"], catalog)

        plocs = cls.plocs_from_ra_dec(ra, dec, wcs).squeeze(0)
        x0_mask = (plocs[:, 0] > 0) & (plocs[:, 0] < height)
        x1_mask = (plocs[:, 1] > 0) & (plocs[:, 1] < width)
        plocs_mask = x0_mask * x1_mask

        galid = galid[plocs_mask]
        plocs = plocs[plocs_mask]

        shear1 = shear1[plocs_mask]
        shear2 = shear2[plocs_mask]
        convergence = convergence[plocs_mask]
        ellip1_lensed = ellip1_lensed[plocs_mask]
        ellip2_lensed = ellip2_lensed[plocs_mask]

        redshift = redshift[plocs_mask]

        mag_mask = mag_mask[plocs_mask]

        nobj = galid.shape[0]
        # TODO: pass existant shear & convergence masks in d
        d = {
            "plocs": plocs.reshape(1, nobj, 2),
            "shear1": shear1.reshape(1, nobj, 1),
            "shear2": shear2.reshape(1, nobj, 1),
            "convergence": convergence.reshape(1, nobj, 1),
            "ellip1_lensed": ellip1_lensed.reshape(1, nobj, 1),
            "ellip2_lensed": ellip2_lensed.reshape(1, nobj, 1),
            "magnitude_cut_mask": mag_mask.reshape(1, nobj, 1),
            "redshift": redshift.reshape(1, nobj, 1),
        }

        return cls(height, width, d), psf_params
