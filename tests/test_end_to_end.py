from bliss.predict import predict
from bliss.train import train


class TestEndToEnd:
    def test_train(self, cfg):
        train(cfg)

    def test_predict(self, cfg):
        predict(cfg)
