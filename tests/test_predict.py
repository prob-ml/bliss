from bliss.predict import predict_sdss


class TestEndToEnd:
    def test_predict(self, cfg):
        the_cfg = cfg.copy()
        the_cfg.predict.plot.show_plot = True
        predict_sdss(the_cfg)
