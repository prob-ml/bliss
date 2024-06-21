from typing import Tuple

import torch
from einops import rearrange
from hydra.utils import instantiate
from pytorch_lightning.utilities import move_data_to_device

from bliss.catalog import FullCatalog
from bliss.surveys.dc2 import DC2, unsqueeze_tile_dict
from case_studies.dc2_cataloging.utils.load_lsst import get_lsst_full_cat


def get_full_cat(
    notebook_cfg, test_img_idx, model_path, lsst_root_dir, device
) -> Tuple[torch.Tensor, FullCatalog, FullCatalog, FullCatalog]:
    dc2: DC2 = instantiate(notebook_cfg.surveys.dc2)
    test_sample = dc2.get_plotting_sample(test_img_idx)
    cur_image_wcs = test_sample["wcs"]
    cur_image_true_full_catalog = test_sample["full_catalog"]
    image_lim = test_sample["image"].shape[1]
    r_band_min_flux = notebook_cfg.encoder.min_flux_for_metrics

    lsst_full_cat = get_lsst_full_cat(
        lsst_root_dir=lsst_root_dir,
        cur_image_wcs=cur_image_wcs,
        image_lim=image_lim,
        r_band_min_flux=r_band_min_flux,
        device=device,
    )

    bliss_encoder = instantiate(notebook_cfg.encoder).to(device=device)
    pretrained_weights = torch.load(model_path, device)["state_dict"]
    bliss_encoder.load_state_dict(pretrained_weights)
    bliss_encoder.eval()

    batch = {
        "tile_catalog": unsqueeze_tile_dict(test_sample["tile_catalog"]),
        "images": rearrange(test_sample["image"], "c h w -> 1 c h w"),
        "background": rearrange(test_sample["background"], "c h w -> 1 c h w"),
        "psf_params": rearrange(test_sample["psf_params"], "h w -> 1 h w"),
    }
    batch = move_data_to_device(batch, device=device)
    bliss_out_dict = bliss_encoder.predict_step(batch, None)
    bliss_full_cat: FullCatalog = bliss_out_dict["mode_cat"].to_full_catalog()

    return test_sample["image"], cur_image_true_full_catalog, bliss_full_cat, lsst_full_cat
