import numpy as np
from mock_tests import mock_predict_sdss_recon

from bliss.surveys.sdss import SloanDigitalSkySurvey as SDSS


class TestSdssReconstruct:
    def test_sdss_reconstruct(self, cfg, decoder):
        # TODO: change to use mock_predict_sdss
        est_full, true_img, true_bg, _, _ = mock_predict_sdss_recon(cfg)

        # reconstruction test only considers r-band image/catalog params
        rcfs = np.array([[94, 1, 12]])
        tile_cat = est_full.to_tile_params(
            tile_slen=cfg.simulator.prior.tile_slen,
            max_sources_per_tile=cfg.simulator.prior.max_sources,
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
