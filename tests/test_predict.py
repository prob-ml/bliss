from bliss.predict import predict_decals, predict_sdss


class TestPredict:
    def test_predict_sdss(self, cfg):
        the_cfg = cfg.copy()
        the_cfg.predict.plot.show_plot = True
        predict_sdss(the_cfg)

    def test_predict_sdss_single_band(self, cfg):
        the_cfg = cfg.copy()
        the_cfg.predict.plot.show_plot = True
        the_cfg.encoder.bands = [2]
        the_cfg.predict.weight_save_path = "${paths.pretrained_models}/single_band_base.pt"
        predict_sdss(the_cfg)

    def test_predict_decals(self, cfg):
        the_cfg = cfg.copy()
        the_cfg.predict.dataset = cfg.surveys.decals
        the_cfg.encoder.bands = [2]
        the_cfg.predict.weight_save_path = "${paths.pretrained_models}/single_band_base.pt"
        predict_decals(the_cfg)
