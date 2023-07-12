import numpy as np
from mock_tests import mock_predict

from bliss.surveys.sdss import SloanDigitalSkySurvey as SDSS


class TestSdssReconstruct:
    def test_sdss_reconstruct(self, cfg, decoder):
        est_tile, true_imgs, true_bgs, _, _ = mock_predict(cfg)

        num_image_units = len(true_imgs)
        assert (
            len(true_bgs) == num_image_units and num_image_units == 1
        ), f"Expected predictions to be made on 1 image unit, got {num_image_units}"
        true_img = true_imgs[next(iter(true_imgs))][0]
        true_bg = true_bgs[next(iter(true_bgs))][0]

        # reconstruction test only considers r-band image/catalog params
        rcfs = np.array([[94, 1, 12]])
        tile_cat = est_tile.to_tile_params(
            tile_slen=cfg.simulator.survey.prior_config.tile_slen,
            max_sources_per_tile=cfg.simulator.survey.prior_config.max_sources,
            ignore_extra_sources=True,
        )
        imgs = decoder.render_images(tile_cat.to("cpu"), rcfs)
        recon_img = imgs[0][0, SDSS.BANDS.index("r")]

        ptc = cfg.encoder.tile_slen * cfg.encoder.tiles_to_crop
        true_img_crop = true_img[SDSS.BANDS.index("r")][ptc:-ptc, ptc:-ptc]
        true_bg_crop = true_bg[SDSS.BANDS.index("r")][ptc:-ptc, ptc:-ptc]
        true_bright = true_img_crop - true_bg_crop

        bright_pix_mask = (recon_img - 100) > 0  # originally 100
        res_bright = recon_img[bright_pix_mask] - true_bright[bright_pix_mask]

        recon_img += true_bg_crop
        res_img = recon_img - true_img_crop

        flux_diff = res_bright.abs().sum()
        flux_sum = true_bright[bright_pix_mask].sum()

        assert ((res_img.abs() / recon_img.sqrt()) > 14).sum() == 0
        assert flux_diff / flux_sum < 0.45
