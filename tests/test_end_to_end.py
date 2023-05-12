from hydra.utils import instantiate

from bliss.generate import generate
from bliss.predict import predict, prepare_image
from bliss.train import train


class TestEndToEnd:
    def test_generate(self, cfg):
        generate(cfg)

    def test_train(self, cfg):
        train(cfg)

    def test_predict(self, cfg):
        sdss = instantiate(cfg.predict.dataset)
        prepare_img = prepare_image(sdss[0]["image"], cfg.predict.device)
        prepare_bg = prepare_image(sdss[0]["background"], cfg.predict.device)
        predict(cfg, prepare_img, prepare_bg)
