from bliss.predict import predict
from bliss.train import train


# TODO: add a test of that we can discover one star with an untrained encoder
# if a GPU is available
# TODO: add some tests of blended stars and galaxies
# TODO: add a test with one tile
# TODO: also test some round-trip reconstructions of images
class TestEndToEnd:
    def test_train(self, cfg):
        train(cfg)

    def test_predict(self, cfg):
        predict(cfg)
