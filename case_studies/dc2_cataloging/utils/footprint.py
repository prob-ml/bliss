import math
from typing import Dict

import torch
from einops import rearrange

from bliss.catalog import TileCatalog


class FootPrint:
    def __init__(self, image: torch.Tensor, params_dict: Dict[str, torch.Tensor]):
        self.image = image
        self.params_dict = params_dict

    @classmethod
    def get_ellipticity(cls, galaxy_disk_q, galaxy_bulge_q, galaxy_beta_radians, galaxy_disk_frac):
        disk_ellipticity = (1 - galaxy_disk_q) / (1 + galaxy_disk_q)
        bulge_ellipticity = (1 - galaxy_bulge_q) / (1 + galaxy_bulge_q)

        total_ellipticity = (
            galaxy_disk_frac * disk_ellipticity + (1 - galaxy_disk_frac) * bulge_ellipticity
        )

        ellipticity1 = total_ellipticity * math.cos(2 * galaxy_beta_radians)
        ellipticity2 = total_ellipticity * math.sin(2 * galaxy_beta_radians)

        return torch.tensor([ellipticity1, ellipticity2])

    @classmethod
    def get_footprints(
        cls,
        full_image: torch.Tensor,
        tile_cat: TileCatalog,
        tile_slen: int,
        footprint_slen: int,
        only_out_boundary_sources=True,
        only_one_source_in_subimage=True,
    ):
        assert footprint_slen % 2 == 0
        _n_bands, full_image_h, full_image_w = full_image.shape
        assert footprint_slen < full_image_h and footprint_slen < full_image_w
        half_footprint_slen = footprint_slen // 2
        full_cat = tile_cat.to_full_catalog(tile_slen)
        assert full_cat["plocs"].shape[0] == 1  # batch_size = 1

        plocs = full_cat["plocs"][0]  # (m, 2)
        if only_out_boundary_sources:
            boundary_ori = torch.tensor(
                [half_footprint_slen, half_footprint_slen], dtype=plocs.dtype, device=plocs.device
            ).view(1, -1)
            boundary_end = torch.tensor(
                [full_image_h - half_footprint_slen, full_image_w - half_footprint_slen],
                dtype=plocs.dtype,
                device=plocs.device,
            ).view(1, -1)
            plocs_mask = ((plocs > boundary_ori) & (plocs < boundary_end)).all(dim=-1)  # (m, )
        else:
            plocs_mask = torch.ones_like(plocs).all(dim=-1)  # (m, )
        if only_one_source_in_subimage:
            boxes_ori = (plocs - half_footprint_slen).unsqueeze(0)  # (1, m, 2)
            boxes_end = (plocs + half_footprint_slen).unsqueeze(0)  # (1, m, 2)
            plocs_ex = rearrange(plocs, "m k -> m 1 k")
            plocs_mask &= ((plocs_ex > boxes_ori) & (plocs_ex < boxes_end)).all(dim=-1).sum(
                dim=-1
            ) == 1  # (m, )

        plocs = plocs[plocs_mask, :]
        n_sources = plocs.shape[0]
        if n_sources == 0:
            return []
        full_cat_dict = {
            k: v[0, plocs_mask, ...] for k, v in full_cat.items() if k != "n_sources"
        }  # (m, k)

        footprints_list = []
        for i in range(n_sources):
            source_plocs = plocs[i]
            footprint_ori = source_plocs - half_footprint_slen
            footprint_end = source_plocs + half_footprint_slen
            image_ori = torch.zeros_like(footprint_ori)
            image_end = (
                torch.tensor(
                    [full_image_h, full_image_w],
                    dtype=footprint_end.dtype,
                    device=footprint_end.device,
                )
                - 1
            )
            footprint_ori = torch.where(footprint_ori < image_ori, image_ori, footprint_ori)
            footprint_end = torch.where(footprint_end > image_end, image_end, footprint_end)
            footprint_image = full_image[
                :,
                math.floor(footprint_ori[0]) : math.floor(footprint_end[0]),
                math.floor(footprint_ori[1]) : math.floor(footprint_end[1]),
            ].clone()
            footprint_params_dict = {k: v[i].clone() for k, v in full_cat_dict.items()}

            if "ellipticity" not in footprint_params_dict:
                footprint_params_dict["ellipticity"] = cls.get_ellipticity(
                    footprint_params_dict["galaxy_disk_q"].item(),
                    footprint_params_dict["galaxy_bulge_q"].item(),
                    footprint_params_dict["galaxy_beta_radians"].item(),
                    footprint_params_dict["galaxy_disk_frac"].item(),
                )

            footprints_list.append(cls(footprint_image, footprint_params_dict))

        return footprints_list
