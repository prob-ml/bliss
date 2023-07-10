import numpy as np
from mock_tests import mock_predict


class TestSdssReconstruct:
    def test_sdss_reconstruct(self, cfg, decoder):
        est_tile, true_img, true_bg, _, _ = mock_predict(cfg)

        # reconstruction test only considers r-band image/catalog params
        rcfs = np.array([[94, 1, 12]])
        tile_cat = est_tile.to_tile_params(
            tile_slen=cfg.simulator.survey.prior_config.tile_slen,
            max_sources_per_tile=cfg.simulator.survey.prior_config.max_sources,
        )
        imgs = decoder.render_images(tile_cat.to("cpu"), rcfs)
        recon_img = imgs[0][0, 2]  # r_band

        ptc = cfg.encoder.tile_slen * cfg.encoder.tiles_to_crop
        true_img_crop = true_img[2][ptc:-ptc, ptc:-ptc]
        true_bg_crop = true_bg[2][ptc:-ptc, ptc:-ptc]
        true_bright = true_img_crop - true_bg_crop

        bright_pix_mask = (recon_img - 100) > 0  # originally 100
        res_bright = recon_img[bright_pix_mask] - true_bright[bright_pix_mask]

        recon_img += true_bg_crop
        res_img = recon_img - true_img_crop

        flux_diff = res_bright.abs().sum()
        flux_sum = true_bright[bright_pix_mask].sum()

        assert ((res_img.abs() / recon_img.sqrt()) > 14).sum() == 0
        assert flux_diff / flux_sum < 0.45
