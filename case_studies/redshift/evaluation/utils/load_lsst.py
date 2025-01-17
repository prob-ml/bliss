import GCRCatalogs
import pandas as pd
import torch

from bliss.catalog import FullCatalog, SourceType


def get_lsst_catalog_tensors_dict(lsst_root_dir: str):
    GCRCatalogs.set_root_dir(lsst_root_dir)
    lsst_catalog_gcr = GCRCatalogs.load_catalog("desc_dc2_run2.2i_dr6_object_with_truth_match")
    lsst_catalog_sub = lsst_catalog_gcr.get_quantities(
        [
            "id_truth",
            "objectId",
            "ra",
            "dec",
            "truth_type",
            "cModelFlux_u",
            "cModelFluxErr_u",
            "cModelFlux_g",
            "cModelFluxErr_g",
            "cModelFlux_r",
            "cModelFluxErr_r",
            "cModelFlux_i",
            "cModelFluxErr_i",
            "cModelFlux_z",
            "cModelFluxErr_z",
            "cModelFlux_y",
            "cModelFluxErr_y",
        ]
    )
    lsst_catalog_df = pd.DataFrame(lsst_catalog_sub)
    lsst_flux_lst = [
        lsst_catalog_df["cModelFlux_g"],
        lsst_catalog_df["cModelFlux_i"],
        lsst_catalog_df["cModelFlux_r"],
        lsst_catalog_df["cModelFlux_u"],
        lsst_catalog_df["cModelFlux_y"],
        lsst_catalog_df["cModelFlux_z"],
    ]
    lsst_flux_tensors_lst = [torch.tensor(flux.values).view(-1, 1) for flux in lsst_flux_lst]
    return {
        "truth_type": torch.tensor(lsst_catalog_df["truth_type"].values).view(-1, 1),
        "flux": torch.cat(lsst_flux_tensors_lst, dim=1),
        "ra": torch.tensor(lsst_catalog_df["ra"].values),
        "dec": torch.tensor(lsst_catalog_df["dec"].values),
    }


def get_lsst_params(
    lsst_catalog_tensors_dict,
    cur_image_wcs,
    image_lim,
):
    lsst_ra = lsst_catalog_tensors_dict["ra"]
    lsst_dec = lsst_catalog_tensors_dict["dec"]
    lsst_pt, lsst_pr = cur_image_wcs.all_world2pix(lsst_ra, lsst_dec, 0)
    lsst_pt = torch.from_numpy(lsst_pt)
    lsst_pr = torch.from_numpy(lsst_pr)

    lsst_plocs = torch.stack((lsst_pr, lsst_pt), dim=-1)
    lsst_source_type = lsst_catalog_tensors_dict["truth_type"]
    lsst_flux = lsst_catalog_tensors_dict["flux"]

    x0_mask = (lsst_plocs[:, 0] > 0) & (lsst_plocs[:, 0] < image_lim)
    x1_mask = (lsst_plocs[:, 1] > 0) & (lsst_plocs[:, 1] < image_lim)
    lsst_x_mask = x0_mask * x1_mask
    # filter r band
    lsst_flux_mask = lsst_flux[:, 2] > 0
    # filter supernova
    lsst_source_mask = (lsst_source_type != 3).squeeze(1)
    lsst_mask = lsst_x_mask * lsst_flux_mask * lsst_source_mask

    lsst_plocs = lsst_plocs[lsst_mask, :]
    lsst_source_type = torch.where(
        lsst_source_type[lsst_mask] == 2, SourceType.STAR, SourceType.GALAXY
    )
    lsst_flux = lsst_flux[lsst_mask, :]

    return lsst_plocs, lsst_source_type, lsst_flux


def get_lsst_full_cat(lsst_root_dir: str, cur_image_wcs, image_lim, r_band_min_flux, device):
    lsst_catalog_tensors_dict = get_lsst_catalog_tensors_dict(lsst_root_dir)
    lsst_plocs, lsst_source_type, lsst_flux = get_lsst_params(
        lsst_catalog_tensors_dict, cur_image_wcs, image_lim
    )
    flux_mask = lsst_flux[:, 2] > r_band_min_flux
    lsst_plocs = lsst_plocs[flux_mask, :]
    lsst_source_type = lsst_source_type[flux_mask]
    lsst_flux = lsst_flux[flux_mask, :]
    lsst_n_sources = torch.tensor([lsst_plocs.shape[0]])

    return FullCatalog(
        height=image_lim,
        width=image_lim,
        d={
            "plocs": lsst_plocs.unsqueeze(0).to(device=device),
            "n_sources": lsst_n_sources.to(device=device),
            "source_type": lsst_source_type.unsqueeze(0).to(device=device),
            "galaxy_fluxes": lsst_flux.unsqueeze(0).to(device=device),
            "star_fluxes": lsst_flux.unsqueeze(0).clone().to(device=device),
        },
    )
