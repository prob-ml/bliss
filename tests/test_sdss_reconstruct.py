import numpy as np
from hydra.utils import instantiate

from bliss.predict import predict_sdss


class TestSdssReconstruct:
    def test_sdss_reconstruct(self, cfg):
        est_tile, true_img, true_bg, _, _ = predict_sdss(cfg)

        # reconstruction test only considers r-band image/catalog params
        decoder_obj = instantiate(cfg.simulator.decoder)
        rcfs = np.array([[94, 1, 12]])
        imgs = decoder_obj.render_images(est_tile.to("cpu"), rcfs)
        recon_img = imgs[0][0, 2]  # r_band

        ptc = cfg.encoder.tile_slen * cfg.encoder.tiles_to_crop
        true_img_crop = true_img[2][ptc:-ptc, ptc:-ptc]
        true_bg_crop = true_bg[2][ptc:-ptc, ptc:-ptc]
        true_bright = true_img_crop - true_bg_crop

        bright_pix_mask = (recon_img - 100) > 0  # originally 100
        recon_img = recon_img.to(cfg.predict.device)
        res_bright = recon_img[bright_pix_mask] - true_bright[bright_pix_mask]

        recon_img += true_bg_crop
        res_img = recon_img - true_img_crop

        flux_diff = res_bright.abs().sum()
        flux_sum = true_bright[bright_pix_mask].sum()

        assert ((res_img.abs() / recon_img.sqrt()) > 14).sum() == 0
        assert flux_diff / flux_sum < 0.45
