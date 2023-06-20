from bliss.generate import generate
from bliss.predict import predict_sdss
from bliss.train import train


class TestEndToEnd:
    def test_generate(self, cfg):
        generate(cfg)

    def test_train(self, cfg):
        train(cfg)

    def test_predict(self, cfg):
        the_cfg = cfg.copy()
        the_cfg.predict.plot.show_plot = True
        predict_sdss(the_cfg)
