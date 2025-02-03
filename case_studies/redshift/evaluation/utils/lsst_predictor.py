# pylint: disable=R0801
import torch

from bliss.catalog import FullCatalog, SourceType, TileCatalog
from bliss.surveys.dc2 import wcs_from_wcs_header_str
from case_studies.dc2_cataloging.utils.load_full_cat import concatenate_tile_dicts
from case_studies.dc2_cataloging.utils.load_lsst import get_lsst_catalog_tensors_dict


class LSSTPredictor:
    def __init__(
        self, lsst_root_dir, r_band_min_flux, tile_slen, max_sources_per_tile, tiles_to_crop
    ) -> None:
        self.lsst_data = get_lsst_catalog_tensors_dict(lsst_root_dir)
        self.r_band_min_flux = r_band_min_flux
        self.tile_slen = tile_slen
        self.max_sources_per_tile = max_sources_per_tile
        self.tiles_to_crop = tiles_to_crop

        self.buffered_wcs_header_str = None
        self.buffered_lsst_plocs = None

    def _predict_one_image(self, wcs_header_str, image_lim, height_index, width_index):
        if wcs_header_str != self.buffered_wcs_header_str:
            lsst_ra = self.lsst_data["ra"]
            lsst_dec = self.lsst_data["dec"]
            cur_image_wcs = wcs_from_wcs_header_str(wcs_header_str)
            lsst_plocs = FullCatalog.plocs_from_ra_dec(lsst_ra, lsst_dec, cur_image_wcs)
            self.buffered_wcs_header_str = wcs_header_str
            self.buffered_lsst_plocs = lsst_plocs
        else:
            lsst_plocs = self.buffered_lsst_plocs

        lsst_source_type = self.lsst_data["type"]
        lsst_flux = self.lsst_data["flux"]

        x0_mask = (lsst_plocs[:, 0] > height_index * image_lim) & (
            lsst_plocs[:, 0] < (height_index + 1) * image_lim
        )
        x1_mask = (lsst_plocs[:, 1] > width_index * image_lim) & (
            lsst_plocs[:, 1] < (width_index + 1) * image_lim
        )
        lsst_x_mask = x0_mask * x1_mask
        # filter r band
        lsst_flux_mask = lsst_flux[:, 2] > self.r_band_min_flux
        lsst_mask = lsst_x_mask * lsst_flux_mask

        lsst_plocs = lsst_plocs[lsst_mask, :] % image_lim
        lsst_source_type = torch.where(
            lsst_source_type[lsst_mask], SourceType.STAR, SourceType.GALAXY
        )
        lsst_flux = lsst_flux[lsst_mask, :]
        lsst_n_sources = torch.tensor([lsst_plocs.shape[0]])

        return FullCatalog(
            height=image_lim,
            width=image_lim,
            d={
                "plocs": lsst_plocs.unsqueeze(0),
                "n_sources": lsst_n_sources,
                "source_type": lsst_source_type.unsqueeze(0),
                "galaxy_fluxes": lsst_flux.unsqueeze(0),
                "star_fluxes": lsst_flux.unsqueeze(0).clone(),
            },
        ).to_tile_catalog(self.tile_slen, self.max_sources_per_tile)

    def predict(self, wcs_header_str_list, image_lim, height_index_list, width_index_list):
        assert len(wcs_header_str_list) == len(height_index_list), "unequal input list size"
        assert len(wcs_header_str_list) == len(width_index_list), "unequal input list size"
        tile_dict_list = []
        predict_input_data = zip(wcs_header_str_list, height_index_list, width_index_list)
        for wcs_header_str, height_index, width_index in predict_input_data:
            tile_dict_list.append(
                self._predict_one_image(wcs_header_str, image_lim, height_index, width_index).data
            )

        merged_tile_dict = concatenate_tile_dicts(tile_dict_list)
        return TileCatalog(self.tile_slen, merged_tile_dict).symmetric_crop(self.tiles_to_crop)
