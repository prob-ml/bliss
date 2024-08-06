from typing import Tuple

import torch
from einops import rearrange
from hydra.utils import instantiate
from pytorch_lightning.utilities import move_data_to_device

from bliss.catalog import FullCatalog, TileCatalog
from bliss.encoder.encoder import Encoder
from bliss.surveys.dc2 import DC2DataModule, split_tensor
from case_studies.dc2_cataloging.utils.load_lsst import get_lsst_full_cat


def concatenate_tile_dicts(tile_dict_list):
    output_tile_cat_dict = {}
    for k in tile_dict_list[0].keys():
        if k not in output_tile_cat_dict:
            output_tile_cat_dict[k] = []
        for tile_dict in tile_dict_list:
            output_tile_cat_dict[k].append(tile_dict[k])

    for k, v in output_tile_cat_dict.items():
        output_tile_cat_dict[k] = torch.cat(v, dim=0)

    return output_tile_cat_dict


def get_full_cat(
    notebook_cfg,
    test_img_idx,
    model_path,
    lsst_root_dir,
    device,
    batch_size=100,
    predict_full_image=False,
) -> Tuple[torch.Tensor, FullCatalog, FullCatalog, FullCatalog]:
    dc2: DC2DataModule = instantiate(notebook_cfg.surveys.dc2)
    test_sample = dc2.get_plotting_sample(test_img_idx)
    cur_image_wcs = test_sample["wcs"]
    image_lim = test_sample["image"].shape[1]
    r_band_min_flux = notebook_cfg.notebook_var.r_band_min_flux
    tile_slen = notebook_cfg.surveys.dc2.tile_slen
    true_tile_dict = DC2DataModule.unsqueeze_tile_dict(test_sample["tile_catalog"])
    true_tile_catalog = TileCatalog(true_tile_dict).filter_by_flux(min_flux=r_band_min_flux)

    lsst_full_cat = get_lsst_full_cat(
        lsst_root_dir=lsst_root_dir,
        cur_image_wcs=cur_image_wcs,
        image_lim=image_lim,
        r_band_min_flux=r_band_min_flux,
    )

    if predict_full_image:
        device = torch.device("cpu")

    bliss_encoder: Encoder = instantiate(notebook_cfg.encoder).to(device=device)
    pretrained_weights = torch.load(model_path, device)["state_dict"]
    bliss_encoder.load_state_dict(pretrained_weights)
    bliss_encoder.eval()

    if predict_full_image:
        batch = {
            "images": rearrange(test_sample["image"], "c h w -> 1 c h w"),
        }
        batch = move_data_to_device(batch, device=device)
        with torch.no_grad():
            bliss_output = bliss_encoder.sample(batch, use_mode=True)
            bliss_full_cat = bliss_output.filter_by_flux(min_flux=r_band_min_flux).to_full_catalog(
                tile_slen
            )
    else:
        split_size = notebook_cfg.surveys.dc2.image_lim[0] // notebook_cfg.surveys.dc2.n_image_split
        image_splits = split_tensor(test_sample["image"], split_size, 1, 2)
        bliss_output_list = []
        for i in range(0, len(image_splits), batch_size):
            batch = {
                "images": torch.stack(image_splits[i : (i + batch_size)]),
            }

            batch = move_data_to_device(batch, device=device)
            with torch.no_grad():
                bliss_output = bliss_encoder.sample(batch, use_mode=True)
                bliss_output = bliss_output.filter_by_flux(min_flux=r_band_min_flux).data
            bliss_output = move_data_to_device(bliss_output, device="cpu")
            bliss_output_list.append(bliss_output)

        d = concatenate_tile_dicts(bliss_output_list)
        n_image_split = notebook_cfg.surveys.dc2.n_image_split
        for k, v in d.items():
            if k != "n_sources":
                d[k] = rearrange(v, "(a b) nth ntw m k -> (a nth) (b ntw) m k", a=n_image_split)
            else:
                d[k] = rearrange(v, "(a b) nth ntw -> (a nth) (b ntw)", a=n_image_split)
        d = DC2DataModule.unsqueeze_tile_dict(d)
        bliss_full_cat = TileCatalog(d).to_full_catalog(tile_slen)

    return (
        test_sample["image"],
        true_tile_catalog.to_full_catalog(tile_slen),
        bliss_full_cat,
        lsst_full_cat,
    )
