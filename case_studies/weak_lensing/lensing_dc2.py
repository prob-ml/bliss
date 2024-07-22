import logging
import sys

import pandas as pd
import torch

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

        self.image_slen = image_slen
        self.bands = self.BANDS
        self.n_bands = len(self.BANDS)

    # _load_image_and_bg_files_list can stay the same

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
        }

        generate_data_wrapper = lambda image_index: self.generate_cached_data(
            image_index,
            **generate_data_input,
        )

        for i in range(n_image):
            generate_data_wrapper(i)

    def load_image_and_catalog(self, image_index, **kwargs):
        image, bg, wcs_header_str = self.read_image_for_bands(image_index)
        wcs = wcs_from_wcs_header_str(wcs_header_str)

        plocs_lim = image[0].shape
        height = plocs_lim[0]
        width = plocs_lim[1]
        full_cat, psf_params = LensingDC2Catalog.from_file(
            kwargs["dc2_cat_path"],
            wcs,
            height,
            width,
            bands=kwargs["bands"],
            n_bands=kwargs["n_bands"],
        )
        tile_cat = full_cat.to_tile_catalog(
            kwargs["tile_slen"],
            kwargs["tile_slen"] ** 2,
        )  # TODO: find the actual max sources with given tile size

        tile_dict = self.squeeze_tile_dict(tile_cat.data)

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

    def generate_cached_data(self, image_index, **kwargs):
        result_dict = self.load_image_and_catalog(image_index, **kwargs)

        image = result_dict["inputs"]["image"]
        bg = result_dict["inputs"]["bg"]
        tile_dict = result_dict["tile_dict"]
        wcs_header_str = result_dict["other_info"]["wcs_header_str"]
        psf_params = result_dict["inputs"]["psf_params"]

        # TODO: interpolation
        shear = tile_dict["shear"]
        convergence = tile_dict["convergence"]
        nonzero_shear_mask = ~torch.all(shear == 0, dim=1)
        nonzero_conv_mask = convergence != 0

        avg_nonzero_convergence = torch.mean(convergence * nonzero_conv_mask, axis=2)
        avg_nonzero_shear = torch.mean(shear * nonzero_shear_mask, axis=2)

        tile_dict["shear"] = avg_nonzero_shear
        tile_dict["convergence"] = avg_nonzero_convergence

        data_to_cache = {
            "tile_catalog": tile_dict,
            "images": image,
            "background": bg,
            "psf_params": psf_params,
            "wcs_header_str": wcs_header_str,
        }

        # Create file name for cached data
        cached_data_file_name = f"cached_data_{image_index:04d}_size_{len(data_to_cache):04d}.pt"

        # Save all data to a single file (no splits)
        with open(kwargs["cached_data_path"] / cached_data_file_name, "wb") as cached_data_file:
            torch.save(data_to_cache, cached_data_file)


class LensingDC2Catalog(DC2FullCatalog):
    @classmethod
    def from_file(cls, cat_path, wcs, height, width, **kwargs):
        catalog = pd.read_pickle(cat_path)

        objid = torch.from_numpy(catalog["id"].values)
        ra = torch.from_numpy(catalog["ra"].values).squeeze()
        dec = torch.from_numpy(catalog["dec"].values).squeeze()

        shear1 = torch.from_numpy(catalog["shear_1"].values).squeeze()
        shear2 = torch.from_numpy(catalog["shear_2"].values).squeeze()
        convergence = torch.from_numpy(catalog["convergence"].values)

        _, psf_params = cls.get_bands_flux_and_psf(kwargs["bands"], catalog)

        plocs = cls.plocs_from_ra_dec(ra, dec, wcs).squeeze(0)
        x0_mask = (plocs[:, 0] > 0) & (plocs[:, 0] < height)
        x1_mask = (plocs[:, 1] > 0) & (plocs[:, 1] < width)
        plocs_mask = x0_mask * x1_mask

        objid = objid[plocs_mask]
        plocs = plocs[plocs_mask]

        shear1 = shear1[plocs_mask]
        shear2 = shear2[plocs_mask]
        convergence = convergence[plocs_mask]
        shear = torch.stack((shear1, shear2), dim=1)

        nobj = objid.shape[0]
        d = {
            "objid": objid.reshape(1, nobj, 1),
            "n_sources": torch.tensor((nobj,)),
            "plocs": plocs.reshape(1, nobj, 2),
            "shear": shear.reshape(1, nobj, 2),
            "convergence": convergence.reshape(1, nobj, 1),
        }

        return cls(height, width, d), psf_params
