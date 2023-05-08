import torch
from hydra.utils import instantiate
from bliss.predict import predict

class TestSdssReconstrust:
    def test_sdss_reconstrst(self, cfg):
        est_full_cat = predict(cfg)
        decoder_obj = instantiate(cfg.simulator.decoder)
        recon_img = decoder_obj.render_images(est_full_cat)[0, 0]
        sdss = instantiate(cfg.predict.dataset)

        true_img = (sdss[0]["image"][ : , :160, :160] - sdss[0]["background"][:, :160, :160])[0] 
        bright_pix_mask = torch.tensor(recon_img - 100) > 0
        res_img = recon_img[bright_pix_mask] - torch.tensor(true_img)[bright_pix_mask]
        flux_diff = res_img.abs().sum()
        flux_sum = torch.tensor(true_img)[bright_pix_mask].sum()
        assert flux_diff / flux_sum < 0.2
