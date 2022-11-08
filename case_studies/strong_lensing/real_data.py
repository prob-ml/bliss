import matplotlib.pyplot as plt
import torch
import numpy as np
import shutil
import os

from bliss.inference import SDSSFrame
from bliss.datasets import sdss
from bliss.inference import reconstruct_scene_at_coordinates
from case_studies.strong_lensing.plots.main import load_models

from astropy.io import fits
from astropy.table import Table

# load models
from hydra import compose, initialize
from hydra.utils import instantiate
from bliss.encoder import Encoder

prob_lenses = [
    .1, .15, .2, .25,
]
min_fluxes = [
    "305", "308", "312", "420", "371", 
]

# for min_flux in min_fluxes:
for i in range(1):
    dir_name = "/home/yppatel/bliss/case_studies/strong_lensing/models"
    lensing_encoder_pts = [os.path.join(dir_name, fn) for fn in os.listdir(dir_name) if fn.startswith("lensing_binary_encoder_") and fn.endswith(".pt")]
    
    for original_fn in lensing_encoder_pts:
        # original_fn = f"/home/yppatel/bliss/case_studies/strong_lensing/models/lensing_binary_encoder_{prob_lens}_{min_flux}.pt"
        new_fn = "/home/yppatel/bliss/case_studies/strong_lensing/models/lensing_binary_encoder.pt"

        if not os.path.exists(original_fn):
            continue

        shutil.copyfile(original_fn, new_fn)

        # load sdss data
        sdss_dir = '/home/yppatel/bliss/data/sdss/'
        pixel_scale = 0.393
        coadd_file = "/home/yppatel/bliss/data/sdss/coadd_catalog_94_1_12.fits"
        frame = SDSSFrame(sdss_dir, pixel_scale, coadd_file)

        device = torch.device('cuda:0')

        with initialize(config_path="config"):
            cfg = compose("config", overrides=[])
            
        enc, dec = load_models(cfg, device)
        bp = enc.border_padding
        torch.cuda.empty_cache()

        # get catalog 
        h, w = bp, bp
        h_end = ((frame.image.shape[2] - 2 * bp) // 4) * 4 + bp #adjustments when using whole frame.
        w_end = ((frame.image.shape[3] - 2 * bp) // 4) * 4 + bp
        coadd_params = frame.get_catalog((h, h_end), (w, w_end))

        dataset_to_params = {
            "horseshoe": {
                "filename": "../../data/sdss/5313/3/117/frame-r-005313-3-0117.fits",
                "x_bounds": [903, 1000],
                "y_bounds": [1003, 1063],
            },
            "8_oclock": {
                "filename": "../../data/sdss/1904/2/187/frame-r-001904-2-0187.fits",
                "x_bounds": [1400, 1475],
                "y_bounds": [810, 860],
            },
            "3_oclock": {
                "filename": "../../data/sdss/2830/2/406/frame-r-002830-2-0406.fits",
                "x_bounds": [925, 1000],
                "y_bounds": [200, 250],
            },
        }

        # cur_dataset = "8_oclock"
        for cur_dataset in dataset_to_params:
            params = dataset_to_params[cur_dataset]

            f = fits.open(params["filename"])[0].data
            img_cp = torch.from_numpy(np.log(np.log(f - f.min() + 2) + 1) * 1600).unsqueeze(0).unsqueeze(0)

            #inference
            with torch.no_grad():
                _, tile_est = reconstruct_scene_at_coordinates(
                    enc,
                    dec,
                    img_cp,
                    frame.background,
                    h_range=(h, h_end),
                    w_range=(w, w_end),
                )
            map_recon = tile_est.to_full_params()
            torch.cuda.empty_cache()

            # prepare inference locs
            plocs = map_recon.plocs.cpu().numpy().squeeze() + bp - 0.5 # plotting adjustment
            coords = frame.wcs.all_pix2world(np.hstack([plocs[:, 1, None], plocs[:, 0, None]]), 0)
            galaxy_bool = map_recon['galaxy_bools'].cpu().numpy().astype(bool).squeeze()
            galaxy_prob = map_recon['galaxy_probs'].cpu().numpy().squeeze()

            image = frame.image.squeeze().numpy()
            lensed_galaxy_bools = map_recon['lensed_galaxy_bools'].cpu().numpy().astype(bool).squeeze()
            lensed_galaxy_probs = map_recon['lensed_galaxy_probs'].cpu().numpy().squeeze()

            plt.rcParams["axes.grid"] = False
            fig, axs = plt.subplots(1, 1, figsize=(8, 6), dpi=120)
            fig = plt.imshow(img_cp[0,0,params["y_bounds"][0]:params["y_bounds"][1], params["x_bounds"][0]:params["x_bounds"][1]])

            lensed_locs = plocs[lensed_galaxy_bools]
            masked_locs = lensed_locs[
                np.logical_and(
                np.logical_and(
                    params["y_bounds"][0] <= lensed_locs[:, 0], lensed_locs[:, 0] <= params["y_bounds"][1],
                ),
                np.logical_and(
                    params["x_bounds"][0] <= lensed_locs[:, 1], lensed_locs[:, 1] <= params["x_bounds"][1],
                ),
                )
            ].copy()

            masked_locs[:, 0] -= params["y_bounds"][0]
            masked_locs[:, 1] -= params["x_bounds"][0]

            print(f"({original_fn}) [{cur_dataset}] -- Correct lenses: {masked_locs.shape}")
            print(f"({original_fn}) [{cur_dataset}] -- All lenses: {lensed_locs.shape}")