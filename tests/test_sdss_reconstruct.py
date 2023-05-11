import torch
from hydra.utils import instantiate
from bliss.predict import predict, prepare_image

class TestSdssReconstrust:
    def test_sdss_reconstrst(self, cfg):
        
        sdss = instantiate(cfg.predict.dataset)
        true_img = sdss[0]["image"][ : , 160:320, 160:320]
        true_bg = sdss[0]["background"][:, 160:320, 160:320]
        est_full = predict(cfg, prepare_image(true_img, cfg.predict.device), 
                           prepare_image(true_bg, cfg.predict.device))
        est_tile = est_full.to_tile_params(cfg.encoder.tile_slen, 
                                           cfg.simulator.prior.max_sources)
        
        decoder_obj = instantiate(cfg.simulator.decoder)
        recon_img = decoder_obj.render_images(est_tile)[0, 0]

        ptc = cfg.encoder.tile_slen * cfg.encoder.tiles_to_crop
        sdss = instantiate(cfg.predict.dataset)
        true_img_crop = true_img[0][ptc:-ptc, ptc:-ptc]
        true_bg_crop = true_bg[0][ptc:-ptc, ptc:-ptc]
        true_bright = true_img_crop - true_bg_crop

        bright_pix_mask = torch.tensor(recon_img - 100) > 0
        res_bright = recon_img[bright_pix_mask] - torch.tensor(true_bright)[bright_pix_mask]

        recon_img += true_bg_crop
        res_img = recon_img - torch.tensor(true_img_crop)

        flux_diff = res_bright.abs().sum()
        flux_sum = torch.tensor(true_bright)[bright_pix_mask].sum()

        assert ((res_img.abs()/recon_img.sqrt())>5).sum() == 0
        assert flux_diff / flux_sum < 0.2
