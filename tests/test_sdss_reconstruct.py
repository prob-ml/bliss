import torch
from hydra.utils import instantiate
from bliss.predict import predict

class TestSdssReconstrust:
    def test_sdss_reconstrst(self, cfg):
        est_tile = predict(cfg).to_tile_params(cfg.encoder.tile_slen, 
                                               cfg.simulator.prior.max_sources)
        decoder_obj = instantiate(cfg.simulator.decoder)
        recon_img = decoder_obj.render_images(est_tile)[0, 0]

        ptc = cfg.encoder.tile_slen * cfg.encoder.tiles_to_crop
        sdss = instantiate(cfg.predict.dataset)
        true_img = sdss[0]["image"][ : , 160:320, 160:320][0][ptc:-ptc, ptc:-ptc]
        true_bg = sdss[0]["background"][:, 160:320, 160:320][0][ptc:-ptc, ptc:-ptc]
        true_bright = true_img - true_bg

        bright_pix_mask = torch.tensor(recon_img - 100) > 0
        res_bright = recon_img[bright_pix_mask] - torch.tensor(true_bright)[bright_pix_mask]

        recon_img += true_bg
        res_img = recon_img - torch.tensor(true_img)

        flux_diff = res_bright.abs().sum()
        flux_sum = torch.tensor(true_bright)[bright_pix_mask].sum()

        assert ((res_img.abs()/recon_img.sqrt())>5).sum() == 0
        assert flux_diff / flux_sum < 0.2
