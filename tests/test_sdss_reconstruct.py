from pathlib import Path

import torch

from bliss.catalog import TileCatalog
from bliss.surveys.sdss import SloanDigitalSkySurvey as SDSS


class TestSdssReconstruct:
    def test_sdss_reconstruct(self, cfg, decoder):
        # return catalog and preds like predict_sdss
        with open(Path(cfg.paths.test_data) / "sdss_preds.pt", "rb") as f:
            data = torch.load(f)
        tile_cat = TileCatalog(cfg.simulator.prior.tile_slen, data["catalog"])

        est_full, true_img, true_bg = tile_cat.to_full_catalog(), data["image"], data["background"]

        # reconstruction test only considers r-band image/catalog params
        rcfs = [(94, 1, 12)]
        tile_cat = est_full.to_tile_catalog(
            tile_slen=cfg.simulator.prior.tile_slen,
            max_sources_per_tile=cfg.simulator.prior.max_sources,
            ignore_extra_sources=True,
        )
        rcfs_indices = torch.zeros(1, dtype=torch.long)
        imgs = decoder.render_images(tile_cat.to("cpu"), rcfs, rcfs_indices)[0]
        imgs = torch.squeeze(imgs, dim=1)
        recon_img = imgs[0, SDSS.BANDS.index("r")]

        true_img_crop = true_img[SDSS.BANDS.index("r"), 0, 4:-4, 4:-4]
        true_bg_crop = true_bg[SDSS.BANDS.index("r"), 0, 4:-4, 4:-4]
        true_bright = true_img_crop - true_bg_crop

        bright_pix_mask = (recon_img - 100) > 0  # originally 100
        res_bright = recon_img[bright_pix_mask] - true_bright[bright_pix_mask]

        recon_img += true_bg_crop
        res_img = recon_img - true_img_crop

        flux_diff = res_bright.abs().sum()
        flux_sum = true_bright[bright_pix_mask].sum()

        assert ((res_img.abs() / recon_img.sqrt()) > 7).sum() == 0
        assert flux_diff / flux_sum < 0.25
